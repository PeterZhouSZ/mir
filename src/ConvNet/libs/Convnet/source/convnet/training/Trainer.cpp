/*
 * C++ Convnet Implementation - Distributed for "Mental Image Retrieval" implementation
 * Copyright (C) 2017-2019  Andreas Ley <mail@andreas-ley.com>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include "Trainer.h"

#include "../Convnet.h"
#include "../layers/InputLayer.h"
#include "TrainingDataSource.h"

#include <cudaUtils/CudaProfilingScope.h>
#include <tools/CPUStopWatch.h>
#include <tools/RasterImage.h>

#include <boost/format.hpp>

#include <iostream>

namespace convnet {

Trainer::Trainer(ConvNet &convnet, TrainingDataSource &dataSource) : m_convnet(convnet), m_dataSource(dataSource)
{
    m_computeStream = convnet.allocateExecutionStream();
//    m_uploadStream = convnet.allocateExecutionStream();


//    m_uploadDoneFence = convnet.allocateSyncFence();
    m_forwardDoneFence = convnet.allocateWaitFence();
    m_backwardDoneFence = convnet.allocateWaitFence();


    m_state = convnet.allocateState(*m_computeStream);
    m_executionWorkspace = convnet.allocateExecutionWorkspace();

    m_inputLayer = m_convnet.getLayer<InputLayer>("input");
    m_stagingBuffer = m_inputLayer->allocateStagingBuffer(*m_computeStream);
}


bool Trainer::learn(unsigned numBatchesPerStep, float stepsize)
{
    Engine::CPUStopWatch stopWatch;

    bool outOfData = false;

    for (unsigned i = 0; i < numBatchesPerStep; i++) {

        {
            AddCudaScopedProfileInterval("Preparing data");
            m_state.clear(*m_computeStream);

            if (!m_dataSource.produceMinibatch(m_stagingBuffer)) {
                outOfData = true;
                break;
            }
            m_backwardDoneFence->waitFor();
            m_inputLayer->swapInStagingBuffer(m_state, m_stagingBuffer);
        }

        {
            AddCudaScopedProfileInterval("Forward");
            m_convnet.feedForward(m_state, *m_executionWorkspace, *m_computeStream, true);

            m_computeStream->insertFence(*m_forwardDoneFence);
        }
        {
            AddCudaScopedProfileInterval("Backward");
            m_convnet.feedBackward(m_state, *m_executionWorkspace, *m_computeStream, true);
            m_computeStream->insertFence(*m_backwardDoneFence);
        }

        {
            AddCudaScopedProfileInterval("Reading loss");

            m_forwardDoneFence->waitFor();
            handleLearningLoss();
        }

    }

    {
        AddCudaScopedProfileInterval("Parameter update step");
        m_convnet.performParameterStep(stepsize, *m_executionWorkspace, *m_computeStream);
    }

    std::cout << "Step took " << stopWatch.getNanoseconds() * 1e-9f << " seconds" << std::endl;

    return !outOfData;
}


void Trainer::run()
{
    
}


void Trainer::runDataset(TrainingDataSource &validationSet, std::function<bool(unsigned, NetworkState&)> dataCallback)
{
    m_backwardDoneFence->waitFor();

    validationSet.restart();
    unsigned numSamples = 0;
    while (true) {
        m_state.clear(*m_computeStream);

        std::cout << "\r Running sample " << numSamples << "                      ";

        if (!validationSet.produceMinibatch(m_stagingBuffer)) {
            break;
        }

        m_inputLayer->swapInStagingBuffer(m_state, m_stagingBuffer);

        {
            AddCudaScopedProfileInterval("Forward");
            m_convnet.feedForward(m_state, *m_executionWorkspace, *m_computeStream, false);

            m_computeStream->insertFence(*m_forwardDoneFence);
        }

        m_forwardDoneFence->waitFor();

        if (!dataCallback(numSamples, m_state)) break;
        
        numSamples += dynamic_cast<convnet::TensorData&>(*m_stagingBuffer.outputs[0]).getValues().getNumInstances();
    }
}

float Trainer::sumLoss(unsigned output)
{
    convnet::TensorData &outputLoss = dynamic_cast<convnet::TensorData&>(*m_state.outputs[output]);

    convnet::MappedTensor mappedLoss = outputLoss.getValues().lock();

    float sum = 0.0f;
    const TensorSize &size = mappedLoss.getSize();
    for (unsigned n = 0; n < mappedLoss.getNumInstances(); n++)
        for (unsigned z = 0; z < size.depth; z++)
            for (unsigned y = 0; y < size.height; y++)
                for (unsigned x = 0; x < size.width; x++) {
                    sum += mappedLoss.get<float>(x, y, z, 0, n);
                }

    outputLoss.getValues().unlock(mappedLoss, true);
    return sum;
}

}
