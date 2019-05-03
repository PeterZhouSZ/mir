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

#include "ClassificationTrainer.h"

#include "../Convnet.h"
#include "../layers/InputLayer.h"
#include "../layers/GANLossLayer.h"
#include "TrainingDataSource.h"
#include "../reporting/HTTPReport.h"
#include "../reporting/TensorRenderer.h"

#include <cudaUtils/CudaProfilingScope.h>
#include <tools/CPUStopWatch.h>

#include <tools/RasterImage.h>
#include <boost/format.hpp>

namespace convnet {

namespace {

std::vector<unsigned> findOutputs(const std::vector<std::string> &names, ConvNet &convnet) {
    std::vector<unsigned> outputs;
    outputs.resize(names.size());
    for (unsigned i = 0; i < names.size(); i++) {
        outputs[i] = convnet.getConnectionList().findOutput(names[i]);
        if (outputs[i] == ~0u)
            throw std::runtime_error(std::string("Could not find output ") + names[i]);
    }
    return outputs;
}

}

ClassificationTrainer::ClassificationTrainer(ConvNet &convnet, const std::vector<std::string> &rngSources, TrainingDataSource &dataSource, std::mt19937 &rng) :
            m_convnet(convnet), m_dataSource(dataSource), m_rngSource(findOutputs(rngSources, convnet), rng)
{
    m_computeStream = convnet.allocateExecutionStream();

    m_forwardDoneFence = convnet.allocateWaitFence();
    m_backwardDoneFence = convnet.allocateWaitFence();

    m_computeStream->insertFence(*m_forwardDoneFence);
    m_computeStream->insertFence(*m_backwardDoneFence);


    m_state = convnet.allocateState(*m_computeStream);
    m_executionWorkspace = convnet.allocateExecutionWorkspace();

    

    m_outputLabel = m_convnet.getConnectionList().findOutput("label");
    m_outputInferredLabel = m_convnet.getConnectionList().findOutput("inferredLabel");
    m_outputRadarData = m_convnet.getConnectionList().findOutput("radar");

    m_outputLoss = m_convnet.getConnectionList().findOutput("loss");

    m_inputLayer = m_convnet.getLayer<InputLayer>("input");
    m_stagingBuffer = m_inputLayer->allocateStagingBuffer(*m_computeStream);
}


bool isPrefixedBy(const std::string &str, const std::string &prefix)
{
    if (str.length() < prefix.length())
        return false;
    for (unsigned i = 0; i < prefix.length(); i++)
        if (str[i] != prefix[i])
            return false;
    return true;
}


bool ClassificationTrainer::learn(unsigned numBatchesPerStep, float stepsize)
{
    Engine::CPUStopWatch stopWatch;

    m_lastLoss = 0.0f;

    unsigned numCorrect = 0;
    unsigned numSamples = 0;

    bool outOfData = false;

    for (unsigned i = 0; i < numBatchesPerStep; i++) {

        m_state.clear(*m_computeStream);

        {
            AddCudaScopedProfileInterval("Produce new data");
            if (!m_dataSource.produceMinibatch(m_stagingBuffer)) {
                outOfData = true;
                break;
            }
            m_rngSource.produceMinibatch(m_stagingBuffer);
        }
        {
            AddCudaScopedProfileInterval("Waiting to swap input");
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
            m_convnet.feedBackwardEx(m_state, *m_executionWorkspace, *m_computeStream,
                [this](Layer &layer)->bool{
                    return !isPrefixedBy(layer.getName(), m_fixedPrefix);
                },
                [this](Layer &layer)->bool{
                    return !isPrefixedBy(layer.getName(), m_fixedPrefix);
                }
            );
            m_computeStream->insertFence(*m_backwardDoneFence);
        }


        {
            AddCudaScopedProfileInterval("Waiting on forward pass to compute loss");
            m_forwardDoneFence->waitFor();
        }
        {
            AddCudaScopedProfileInterval("Computing loss");
            // reconstruction error
            {
                AddCudaScopedProfileInterval("Computing reconstruction error");
                convnet::TensorData &loss = dynamic_cast<convnet::TensorData&>(*m_state.outputs[m_outputLoss]);
                convnet::MappedTensor mappedLoss = loss.getValues().lock();

                float reconstructionError = 0.0f;
                for (unsigned n = 0; n < mappedLoss.getNumInstances(); n++)
                    for (unsigned i = 0; i < mappedLoss.getSize().numElements(); i++) {
                        reconstructionError += mappedLoss.data<float>(n)[i];
                    }

                m_lastLoss += reconstructionError;

                loss.getValues().unlock(mappedLoss, false);
            }
            {
                AddCudaScopedProfileInterval("Computing label error");

                convnet::TensorData &inputLabel = dynamic_cast<convnet::TensorData&>(*m_state.outputs[m_outputLabel]);
                convnet::TensorData &outputLabel = dynamic_cast<convnet::TensorData&>(*m_state.outputs[m_outputInferredLabel]);

                convnet::MappedTensor mappedInputLabel = inputLabel.getValues().lock();
                convnet::MappedTensor mappedOutputLabel = outputLabel.getValues().lock();


                int ox = (mappedInputLabel.getSize().width - mappedOutputLabel.getSize().width)/2;
                int oy = (mappedInputLabel.getSize().height - mappedOutputLabel.getSize().height)/2;
                int oz = (mappedInputLabel.getSize().depth - mappedOutputLabel.getSize().depth)/2;

                const TensorSize &size = mappedOutputLabel.getSize();
                for (unsigned n = 0; n < mappedOutputLabel.getNumInstances(); n++)
                    for (unsigned z = 0; z < size.depth; z++)
                        for (unsigned y = 0; y < size.height; y++)
                            for (unsigned x = 0; x < size.width; x++) {
                                int label = mappedInputLabel.get<float>(ox+x, oy+y, oz+z, 0, n);
                                if (label == 0xFF) continue;
                                numSamples++;
                                float maxOut = 0;
                                for (unsigned c = 1; c < mappedOutputLabel.getSize().channels; c++)
                                    if (mappedOutputLabel.get<float>(x, y, z, c, n) > mappedOutputLabel.get<float>(x, y, z, maxOut, n))
                                        maxOut = c;

                                if (maxOut == label)
                                    numCorrect++;                                
                            }

                inputLabel.getValues().unlock(mappedInputLabel, true);
                outputLabel.getValues().unlock(mappedOutputLabel, true);
            }
            
        }

    }
    if (numSamples > 0) {
        m_lastAccuracy = numCorrect / (float) numSamples;
        m_lastLoss = m_lastLoss / (float) numSamples;
    }
    std::cout << "m_lastLoss " << m_lastLoss << std::endl;
    std::cout << "m_lastAccuracy " << m_lastAccuracy << std::endl;

    m_convnet.performParameterStepEx(stepsize, *m_executionWorkspace, *m_computeStream,
        [this](LearnedParameters &params)->bool{
            return !isPrefixedBy(params.getName(), m_fixedPrefix);
        }
    );


    std::cout << "Step took " << stopWatch.getNanoseconds() * 1e-9f << " seconds" << std::endl;

    return !outOfData;
}


void ClassificationTrainer::debugOutputTest(TrainingDataSource &validationSet, const std::string &prefix, unsigned N)
{
    struct Img {
        RasterImage imageData;
        RasterImage reconstructedImageData[5];
        RasterImage tensorData[5];
    };


    std::vector<Img> images;

    images.resize(N);

    validationSet.restart();

    unsigned numSamples = 0;

    while (true) {
        m_state.clear(*m_computeStream);

        std::cout << "\r Running validation sample " << numSamples << "                      ";

        if (!validationSet.produceMinibatch(m_stagingBuffer)) {
            break;
        }
        m_rngSource.produceMinibatch(m_stagingBuffer);
        {
            TensorData &radar = dynamic_cast<TensorData&>(*m_stagingBuffer.outputs[m_outputRadarData]);
            TensorData &label = dynamic_cast<TensorData&>(*m_stagingBuffer.outputs[m_outputLabel]);

            const unsigned numInstances = radar.getValues().getNumInstances();

            MappedTensor mappedRadar = radar.getValues().lock();
            MappedTensor mappedLabel = label.getValues().lock();

            for (unsigned n = 0; n < numInstances; n++) {
                if (numSamples + n >= images.size()) break;

                images[numSamples+n].imageData.resize(mappedRadar.getSize().width*2, mappedRadar.getSize().height);

                for (unsigned y = 0; y < mappedRadar.getSize().height; y++)
                    for (unsigned x = 0; x < mappedRadar.getSize().width; x++) {
                        
                        int label = mappedLabel.get<float>(x, y, 0, 0, n);

const unsigned labelColor_R[] = {     0,    128,    0,   255,    128  };                        
const unsigned labelColor_G[] = {     0,   255,  255,     0,    128  };
const unsigned labelColor_B[] = {   255,     0,   128,     0,    128  };
                        
                        int r = 0;
                        int g = 0;
                        int b = 0;
                        if (label != 0xFF) {
                            r = labelColor_R[label];
                            g = labelColor_G[label];
                            b = labelColor_B[label];
                        }

                        images[numSamples+n].imageData.getData()[x+y*mappedRadar.getSize().width*2] =
                                            (r << 0) |
                                            (g << 8) |
                                            (b << 16) |
                                            (0xFF << 24);

                        r = std::min(std::max<int>(mappedRadar.get<float>(x, y, 0, 0, n) * 127.0f/3 + 127.0f, 0), 255);
                        g = std::min(std::max<int>(mappedRadar.get<float>(x, y, 0, 1, n) * 127.0f/3 + 127.0f, 0), 255);
                        b = 128;

                        images[numSamples+n].imageData.getData()[mappedRadar.getSize().width+x+y*mappedRadar.getSize().width*2] =
                                            (r << 0) |
                                            (g << 8) |
                                            (b << 16) |
                                            (0xFF << 24);
                    }

            }


            radar.getValues().unlock(mappedRadar);
            label.getValues().unlock(mappedLabel);
        }


        m_backwardDoneFence->waitFor();
        m_inputLayer->swapInStagingBuffer(m_state, m_stagingBuffer);

        {
            AddCudaScopedProfileInterval("Forward");
            m_convnet.feedForward(m_state, *m_executionWorkspace, *m_computeStream, false);

            m_computeStream->insertFence(*m_forwardDoneFence);
        }

        m_forwardDoneFence->waitFor();

        auto renderImgData = [&](unsigned output, unsigned rasterImgIndex) {

            TensorData &data = dynamic_cast<TensorData&>(*m_state.outputs[output]);

            const unsigned numInstances = data.getValues().getNumInstances();

            MappedTensor mappedData = data.getValues().lock();

            for (unsigned n = 0; n < numInstances; n++) {
                if (numSamples + n >= images.size()) break;

                images[numSamples+n].reconstructedImageData[rasterImgIndex].resize(mappedData.getSize().width, mappedData.getSize().height);

                for (unsigned y = 0; y < mappedData.getSize().height; y++)
                    for (unsigned x = 0; x < mappedData.getSize().width; x++) {

                        int r = std::min(std::max<int>(mappedData.get<float>(x, y, 0, 0, n) * 127.0f/3 + 127.0f, 0), 255);
                        int g = std::min(std::max<int>(mappedData.get<float>(x, y, 0, 1, n) * 127.0f/3 + 127.0f, 0), 255);
                        int b = std::min(std::max<int>(mappedData.get<float>(x, y, 0, 2, n) * 127.0f/3 + 127.0f, 0), 255);

                        images[numSamples+n].reconstructedImageData[rasterImgIndex].getData()[x+y*mappedData.getSize().width] =
                                            (r << 0) |
                                            (g << 8) |
                                            (b << 16) |
                                            (0xFF << 24);

                    }

            }

            data.getValues().unlock(mappedData);
        };
        auto renderLabelData = [&](unsigned output, unsigned rasterImgIndex) {

            TensorData &data = dynamic_cast<TensorData&>(*m_state.outputs[output]);

            const unsigned numInstances = data.getValues().getNumInstances();

            MappedTensor mappedData = data.getValues().lock();

            for (unsigned n = 0; n < numInstances; n++) {
                if (numSamples + n >= images.size()) break;

                images[numSamples+n].reconstructedImageData[rasterImgIndex].resize(mappedData.getSize().width, mappedData.getSize().height);

                for (unsigned y = 0; y < mappedData.getSize().height; y++)
                    for (unsigned x = 0; x < mappedData.getSize().width; x++) {
                        int label = 0;
                        for (unsigned i = 1; i < mappedData.getSize().channels; i++)
                            if (mappedData.get<float>(x, y, 0, i, n) > mappedData.get<float>(x, y, 0, label, n))
                                label = i;


const unsigned labelColor_R[] = {     0,    128,    0,   255,    128  };                        
const unsigned labelColor_G[] = {     0,   255,  255,     0,    128  };
const unsigned labelColor_B[] = {   255,     0,   128,     0,    128  };
                        
                        int r = 0;
                        int g = 0;
                        int b = 0;
                        if (label != 0xFF) {
                            r = labelColor_R[label];
                            g = labelColor_G[label];
                            b = labelColor_B[label];
                        }
                        
                        images[numSamples+n].reconstructedImageData[rasterImgIndex].getData()[x+y*mappedData.getSize().width] =
                                            (r << 0) |
                                            (g << 8) |
                                            (b << 16) |
                                            (0xFF << 24);

                    }

            }

            data.getValues().unlock(mappedData);
        };
        if (m_outputInferredLabel != ~0u)
            renderLabelData(m_outputInferredLabel, 0);


        auto renderTensorData = [&](unsigned output, unsigned rasterImgIndex) {

            TensorData &data = dynamic_cast<TensorData&>(*m_state.outputs[output]);

            const unsigned numInstances = data.getValues().getNumInstances();

            MappedTensor mappedData = data.getValues().lock();

            for (unsigned n = 0; n < numInstances; n++) {
                if (numSamples + n >= images.size()) break;

                images[numSamples+n].tensorData[rasterImgIndex].resize(mappedData.getSize().width, mappedData.getSize().height);

                for (unsigned y = 0; y < mappedData.getSize().height; y++)
                    for (unsigned x = 0; x < mappedData.getSize().width; x++) {

                        int r = std::min(std::max<int>(mappedData.get<float>(x, y, 0, 0, n) * 127.0f/3 + 127.0f, 0), 255);
                        int g = mappedData.getSize().channels>1?std::min(std::max<int>(mappedData.get<float>(x, y, 0, 1, n) * 127.0f/3 + 127.0f, 0), 255):128;
                        int b = mappedData.getSize().channels>2?std::min(std::max<int>(mappedData.get<float>(x, y, 0, 2, n) * 127.0f/3 + 127.0f, 0), 255):128;

                        images[numSamples+n].tensorData[rasterImgIndex].getData()[x+y*mappedData.getSize().width] =
                                            (r << 0) |
                                            (g << 8) |
                                            (b << 16) |
                                            (0xFF << 24);

                    }

            }

            data.getValues().unlock(mappedData);
        };
        if (m_outputLoss != ~0u)
            renderTensorData(m_outputLoss, 0);
        /*
        if (m_convnet.getConnectionList().findOutput("realLoss") != ~0u)
            renderTensorData(m_convnet.getConnectionList().findOutput("realLoss"), 1);
        if (m_convnet.getConnectionList().findOutput("level3_latent_code") != ~0u)
            renderTensorData(m_convnet.getConnectionList().findOutput("level3_latent_code"), 2);
        if (m_convnet.getConnectionList().findOutput("level4_latent_code") != ~0u)
            renderTensorData(m_convnet.getConnectionList().findOutput("level4_latent_code"), 3);
        if (m_convnet.getConnectionList().findOutput("level5_latent_code") != ~0u)
            renderTensorData(m_convnet.getConnectionList().findOutput("level5_latent_code"), 4);
        */

        numSamples += dynamic_cast<convnet::TensorData&>(*m_stagingBuffer.outputs[0]).getValues().getNumInstances();
        if (numSamples >= images.size()) break;
    }

    for (unsigned i = 0; i < images.size(); i++) {
        if (images[i].imageData.getWidth() != 0)
            images[i].imageData.writeToFile((boost::format("%s_%04d.png") % prefix % i).str().c_str());
        if (images[i].reconstructedImageData[0].getWidth() != 0)
            images[i].reconstructedImageData[0].writeToFile((boost::format("%s_%04d_R.png") % prefix % i).str().c_str());
        /*
        if (images[i].reconstructedImageData[1].getWidth() != 0)
            images[i].reconstructedImageData[1].writeToFile((boost::format("%s_%04d_R_1.png") % prefix % i).str().c_str());
        if (images[i].reconstructedImageData[2].getWidth() != 0)
            images[i].reconstructedImageData[2].writeToFile((boost::format("%s_%04d_R_2.png") % prefix % i).str().c_str());
        if (images[i].reconstructedImageData[3].getWidth() != 0)
            images[i].reconstructedImageData[3].writeToFile((boost::format("%s_%04d_R_3.png") % prefix % i).str().c_str());
        if (images[i].reconstructedImageData[4].getWidth() != 0)
            images[i].reconstructedImageData[4].writeToFile((boost::format("%s_%04d_R_4.png") % prefix % i).str().c_str());
        if (images[i].reconstructedImageData[5].getWidth() != 0)
            images[i].reconstructedImageData[5].writeToFile((boost::format("%s_%04d_R_5.png") % prefix % i).str().c_str());
        */

        if (images[i].tensorData[0].getWidth() != 0)
            images[i].tensorData[0].writeToFile((boost::format("%s_%04d_T_0.png") % prefix % i).str().c_str());
        /*
        if (images[i].tensorData[1].getWidth() != 0)
            images[i].tensorData[1].writeToFile((boost::format("%s_%04d_T_1.png") % prefix % i).str().c_str());
        /*
        if (images[i].tensorData[2].getWidth() != 0)
            images[i].tensorData[2].writeToFile((boost::format("%s_%04d_T_2.png") % prefix % i).str().c_str());
        if (images[i].tensorData[3].getWidth() != 0)
            images[i].tensorData[3].writeToFile((boost::format("%s_%04d_T_3.png") % prefix % i).str().c_str());
        if (images[i].tensorData[4].getWidth() != 0)
            images[i].tensorData[4].writeToFile((boost::format("%s_%04d_T_4.png") % prefix % i).str().c_str());
        if (images[i].tensorData[5].getWidth() != 0)
            images[i].tensorData[5].writeToFile((boost::format("%s_%04d_T_5.png") % prefix % i).str().c_str());
        */
    }
}


void ClassificationTrainer::runValidation(TrainingDataSource &validationSet, float &loss, float &accuracy, Eigen::MatrixXf &confusionMatrix, std::vector<unsigned> &totalPerClass, unsigned border)
{
    unsigned numClasses = m_convnet.getConnectionList().outputs[m_outputInferredLabel].size.channels;

    validationSet.restart();
    
    unsigned numCorrect = 0;
    totalPerClass.clear();
    totalPerClass.resize(numClasses, 0);
    confusionMatrix.resize(numClasses, numClasses);
    confusionMatrix.setZero();
    
    unsigned numSamples = 0;
    
    loss = 0.0f;

    while (true) {
        m_state.clear(*m_computeStream);

        std::cout << "\r Running validation sample " << numSamples << "                      ";

        if (!validationSet.produceMinibatch(m_stagingBuffer)) {
            break;
        }

        m_backwardDoneFence->waitFor();
        m_inputLayer->swapInStagingBuffer(m_state, m_stagingBuffer);

        {
            AddCudaScopedProfileInterval("Forward");
            m_convnet.feedForward(m_state, *m_executionWorkspace, *m_computeStream, false);

            m_computeStream->insertFence(*m_forwardDoneFence);
        }

        m_forwardDoneFence->waitFor();
        {
            AddCudaScopedProfileInterval("Computing reconstruction error");
            convnet::TensorData &lossData = dynamic_cast<convnet::TensorData&>(*m_state.outputs[m_outputLoss]);
            convnet::MappedTensor mappedLoss = lossData.getValues().lock();

            float reconstructionError = 0.0f;
            for (unsigned n = 0; n < mappedLoss.getNumInstances(); n++)
                for (unsigned i = 0; i < mappedLoss.getSize().numElements(); i++) {
                    reconstructionError += mappedLoss.data<float>(n)[i];
                }

            loss += reconstructionError;

            lossData.getValues().unlock(mappedLoss, false);
        }
        {
            AddCudaScopedProfileInterval("Computing label error");

            convnet::TensorData &inputLabel = dynamic_cast<convnet::TensorData&>(*m_state.outputs[m_outputLabel]);
            convnet::TensorData &outputLabel = dynamic_cast<convnet::TensorData&>(*m_state.outputs[m_outputInferredLabel]);

            convnet::MappedTensor mappedInputLabel = inputLabel.getValues().lock();
            convnet::MappedTensor mappedOutputLabel = outputLabel.getValues().lock();


            int ox = (mappedInputLabel.getSize().width - mappedOutputLabel.getSize().width)/2;
            int oy = (mappedInputLabel.getSize().height - mappedOutputLabel.getSize().height)/2;
            int oz = (mappedInputLabel.getSize().depth - mappedOutputLabel.getSize().depth)/2;

            const TensorSize &size = mappedOutputLabel.getSize();
            for (unsigned n = 0; n < mappedOutputLabel.getNumInstances(); n++)
                for (unsigned z = 0; z < size.depth; z++)
                    for (unsigned y = 0; y < size.height; y++)
                        for (unsigned x = 0; x < size.width; x++) {
                            int label = mappedInputLabel.get<float>(ox+x, oy+y, oz+z, 0, n);
                            if (label == 0xFF) continue;
                            numSamples++;
                            totalPerClass[label]++;

                            float maxOut = 0;
                            for (unsigned c = 1; c < mappedOutputLabel.getSize().channels; c++)
                                if (mappedOutputLabel.get<float>(x, y, z, c, n) > mappedOutputLabel.get<float>(x, y, z, maxOut, n))
                                    maxOut = c;

                            confusionMatrix(maxOut, label) += 1.0f;
                            
                            if (maxOut == label)
                                numCorrect++;                                
                        }

            inputLabel.getValues().unlock(mappedInputLabel, true);
            outputLabel.getValues().unlock(mappedOutputLabel, true);
        }
    }

    
    accuracy = numCorrect / (float)numSamples;
    loss = loss / (float)numSamples;
    
}

void ClassificationTrainer::report(TrainingDataSource &validationSet, HTTPReport &report, const std::string &prefix,
            std::vector<std::vector<unsigned>> inputChannelSelection,
            std::vector<unsigned> outputChannelSelection)
{
    validationSet.restart();

    const unsigned numTotalSamples = 16;
    unsigned numSamples = 0;
    
    std::vector<unsigned> instancesToRender;
    for (unsigned i = 0; i < m_convnet.getNumInstances(); i++)
        instancesToRender.push_back(i);
         
    while (true) {
        m_state.clear(*m_computeStream);

        std::cout << "\r Running validation sample " << numSamples << "                      ";

        if (!validationSet.produceMinibatch(m_stagingBuffer)) {
            break;
        }
        m_rngSource.produceMinibatch(m_stagingBuffer);
        {
            TensorRenderer tensorRenderer;
            
            for (unsigned i = 0; i < m_stagingBuffer.outputs.size(); i++) {
                TensorData &tensorData = dynamic_cast<TensorData&>(*m_stagingBuffer.outputs[i]);
                
                if (!inputChannelSelection.empty())
                    tensorRenderer.channelSelection = inputChannelSelection[i];
                
                std::vector<RasterImage> imgs = tensorRenderer.renderTensors(tensorData.getValues(), instancesToRender);
                for (unsigned j = 0; j < imgs.size(); j++)
                    report.putImage(imgs[j], (boost::format("imgs/%s_classification_training_inputs_%d_%d.png") % prefix % i % (numSamples+j)).str());
            }
        }


        m_backwardDoneFence->waitFor();
        m_inputLayer->swapInStagingBuffer(m_state, m_stagingBuffer);

        {
            AddCudaScopedProfileInterval("Forward");
            m_convnet.feedForward(m_state, *m_executionWorkspace, *m_computeStream, false);

            m_computeStream->insertFence(*m_forwardDoneFence);
        }

        m_forwardDoneFence->waitFor();

        if (m_outputInferredLabel != ~0u) {
            TensorData &tensorData = dynamic_cast<TensorData&>(*m_state.outputs[m_outputInferredLabel]);
            TensorRenderer tensorRenderer;
            
            tensorRenderer.channelSelection = outputChannelSelection;
            
            std::vector<RasterImage> imgs = tensorRenderer.renderTensors(tensorData.getValues(), instancesToRender);
            for (unsigned j = 0; j < imgs.size(); j++)
                report.putImage(imgs[j], (boost::format("imgs/%s_classification_training_outputs_%d.png") % prefix % (numSamples+j)).str());
        }



        numSamples += m_convnet.getNumInstances();
        if (numSamples >= numTotalSamples) break;
    }

    std::stringstream site;
    site << "<!DOCTYPE html>" << std::endl
         << "<html>" << std::endl
         << "<head>" << std::endl
         << "<meta charset=\"UTF-8\">" << std::endl
         << "<title>Classification images " << prefix << "</title>" << std::endl
         << "</head>" << std::endl

         << "<body>" << std::endl;
      
    site 
        << "<table>" << std::endl
        << "  <tr>" << std::endl;
     
    for (unsigned i = 0; i < m_stagingBuffer.outputs.size(); i++)
        site << "    <th>Input " << i << "</th>" << std::endl;
    
    site 
        << "    <th>Output</th>" << std::endl
        << "  </tr>" << std::endl;
        
    for (unsigned j = 0; j < numTotalSamples; j++) {
        site 
            << "  <tr>" << std::endl;
        for (unsigned i = 0; i < m_stagingBuffer.outputs.size(); i++)
            site << "    <td><img src=\"imgs/"<<prefix<<"_classification_training_inputs_" << i << "_" << j << ".png\"/></td>" << std::endl;
        site << "    <td><img src=\"imgs/"<<prefix<<"_classification_training_outputs_" << j << ".png\"/></td>" << std::endl;
        site 
            << "  </tr>" << std::endl;
    }


    site
         << "</table>" << std::endl
         << "</body>" << std::endl
         << "</html>" << std::endl;
    
    report.putText(site, prefix+"_classificationTraining.html");
}


}

