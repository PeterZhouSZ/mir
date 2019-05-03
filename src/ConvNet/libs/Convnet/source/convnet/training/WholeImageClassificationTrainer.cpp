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

#include "WholeImageClassificationTrainer.h"

#include "../Convnet.h"
#include "../layers/InputLayer.h"
#include "TrainingDataSource.h"

#include "../reporting/State2Html.h"
#include "../reporting/HTTPReport.h"

#include <cudaUtils/CudaProfilingScope.h>
#include <tools/CPUStopWatch.h>
#include <tools/RasterImage.h>

#include <boost/format.hpp>
#include <iostream>

namespace convnet {

WholeImageClassificationTrainer::WholeImageClassificationTrainer(ConvNet &convnet, TrainingDataSource &dataSource) : m_convnet(convnet), m_dataSource(dataSource)
{
    m_computeStream = convnet.allocateExecutionStream();
//    m_uploadStream = convnet.allocateExecutionStream();


//    m_uploadDoneFence = convnet.allocateSyncFence();
    m_forwardDoneFence = convnet.allocateWaitFence();
    m_backwardDoneFence = convnet.allocateWaitFence();


    m_state = convnet.allocateState(*m_computeStream);
    m_executionWorkspace = convnet.allocateExecutionWorkspace();

    m_outputInputLabels = m_convnet.getConnectionList().findOutput("label");
    m_outputLabels = m_convnet.getConnectionList().findOutput("inferredLabel");
    m_outputLoss = m_convnet.getConnectionList().findOutput("loss");


    m_inputLayer = m_convnet.getLayer<InputLayer>("input");
    m_stagingBuffer = m_inputLayer->allocateStagingBuffer(*m_computeStream);
}


bool WholeImageClassificationTrainer::learn(unsigned numBatchesPerStep, float stepsize)
{
    Engine::CPUStopWatch stopWatch;

    m_lastBatchLabelError = 0.0f;

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
            convnet::TensorData &outputLoss = dynamic_cast<convnet::TensorData&>(*m_state.outputs[m_outputLoss]);

            convnet::MappedTensor mappedLoss = outputLoss.getValues().lock();

            float labelError = 0.0f;
            const TensorSize &size = mappedLoss.getSize();
            for (unsigned n = 0; n < mappedLoss.getNumInstances(); n++)
                for (unsigned z = 0; z < size.depth; z++)
                    for (unsigned y = 0; y < size.height; y++)
                        for (unsigned x = 0; x < size.width; x++) {
                            labelError += mappedLoss.get<float>(x, y, z, 0, n);
                        }

            outputLoss.getValues().unlock(mappedLoss, true);

            m_lastBatchLabelError += labelError;
        }

    }
    std::cout << "m_lastBatchLabelError " << m_lastBatchLabelError << std::endl;

    {
        AddCudaScopedProfileInterval("Parameter update step");
        m_convnet.performParameterStep(stepsize, *m_executionWorkspace, *m_computeStream);
    }

    std::cout << "Step took " << stopWatch.getNanoseconds() * 1e-9f << " seconds" << std::endl;

    return !outOfData;
}

void WholeImageClassificationTrainer::runValidation(TrainingDataSource &validationSet, std::vector<float> &accuracyVsRank, Eigen::MatrixXf *confusion, Eigen::MatrixXf *confusion_top3)
{
    validationSet.restart();
    accuracyVsRank.clear();
    unsigned numLabels = m_convnet.getConnectionList().outputs[m_outputLabels].size.channels;
    accuracyVsRank.resize(numLabels, 0.0f);

    unsigned accuracyVsRankDenom = 0;

    unsigned numSamples = 0;

    while (true) {
        m_state.clear(*m_computeStream);

//        std::cout << "\r Running validation sample " << numSamples << "                      ";
        std::cout << "Running validation sample " << numSamples << "                      ";

        if (!validationSet.produceMinibatch(m_stagingBuffer)) {
            break;
        }
        numSamples += dynamic_cast<convnet::TensorData&>(*m_stagingBuffer.outputs[0]).getValues().getNumInstances();

        m_backwardDoneFence->waitFor();
        m_inputLayer->swapInStagingBuffer(m_state, m_stagingBuffer);

        {
            AddCudaScopedProfileInterval("Forward");
            m_convnet.feedForward(m_state, *m_executionWorkspace, *m_computeStream, false);

            m_computeStream->insertFence(*m_forwardDoneFence);
        }

        m_forwardDoneFence->waitFor();

        {
            AddCudaScopedProfileInterval("Computing label error");

            convnet::TensorData &inputLabel = dynamic_cast<convnet::TensorData&>(*m_state.outputs[m_outputInputLabels]);
            convnet::TensorData &outputLabel = dynamic_cast<convnet::TensorData&>(*m_state.outputs[m_outputLabels]);

            convnet::MappedTensor mappedInputLabel = inputLabel.getValues().lock();
            convnet::MappedTensor mappedOutputLabel = outputLabel.getValues().lock();


            int ox = (mappedInputLabel.getSize().width - mappedOutputLabel.getSize().width)/2;
            int oy = (mappedInputLabel.getSize().height - mappedOutputLabel.getSize().height)/2;
            int oz = (mappedInputLabel.getSize().depth - mappedOutputLabel.getSize().depth)/2;

            std::vector<float> exps;

            const TensorSize &size = mappedOutputLabel.getSize();
            for (unsigned n = 0; n < mappedOutputLabel.getNumInstances(); n++)
                for (unsigned z = 0; z < size.depth; z++)
                    for (unsigned y = 0; y < size.height; y++)
                        for (unsigned x = 0; x < size.width; x++) {
                            int label = mappedInputLabel.get<float>(ox+x, oy+y, oz+z, 0, n);
                            if (label == 0xFF) continue;

                            float maxOut = mappedOutputLabel.get<float>(x, y, z, 0, n);
                            for (unsigned c = 0; c < mappedOutputLabel.getSize().channels; c++)
                                maxOut = std::max(maxOut, mappedOutputLabel.get<float>(x, y, z, c, n));

                            exps.resize(mappedOutputLabel.getSize().channels);
                            float denom = 0.0f;
                            for (unsigned c = 0; c < mappedOutputLabel.getSize().channels; c++) {
                                exps[c] = std::exp(mappedOutputLabel.get<float>(x, y, z, c, n) - maxOut);
                                denom += exps[c];
                            }

                            std::vector<std::pair<float, unsigned>> results;
                            for (unsigned c = 0; c < mappedOutputLabel.getSize().channels; c++) {
                                results.push_back({1.0f - exps[c] / denom, c});
                            }

                            std::sort(results.begin(), results.end());
                            
//							std::cout << "Classification : " << label << "  " << results.front().second << std::endl;
                            if (confusion != nullptr) {
                                (*confusion)(label, results.front().second) += 1.0f;
                            }
							if (confusion_top3 != nullptr) {
                                (*confusion_top3)(label, results[0].second) += 1.0f;
                                (*confusion_top3)(label, results[1].second) += 1.0f;
                                (*confusion_top3)(label, results[2].second) += 1.0f;
							}

                            for (unsigned j = 0; j < numLabels; j++) {
                                if (results[j].second == label) {
                                    for (unsigned k = j; k < numLabels; k++) {
                                        accuracyVsRank[k] += 1.0f;
                                    }
                                    break;
                                }
                            }
                            accuracyVsRankDenom++;
                        }

            inputLabel.getValues().unlock(mappedInputLabel, true);
            outputLabel.getValues().unlock(mappedOutputLabel, true);
        }
    }

    for (auto &f : accuracyVsRank)
        f /= accuracyVsRankDenom;
}

void WholeImageClassificationTrainer::report(TrainingDataSource &validationSet, const std::string &prefix, HTTPReport &report)
{
    validationSet.restart();
    
    State2Html state2html(prefix);
    state2html.addDefaultOutput(
        m_convnet.getConnectionList().findOutput("data"),
        "data"
    );
    state2html.addDefaultOutput(
        m_convnet.getConnectionList().findOutput("label"),
        "label"
    );
    state2html.addDefaultOutput(
        m_convnet.getConnectionList().findOutput("inferredLabel"),
        "inferredLabel"
    );
    
    state2html.getOutputs()[0].tensorRenderer.upsample = 4;
    state2html.getOutputs()[1].tensorRenderer.upsample = 20;
    state2html.getOutputs()[2].tensorRenderer.upsample = 20;
    
    
    std::stringstream site;
    site
        << "<!DOCTYPE html>" << std::endl
        << "<html>" << std::endl
        << "<head>" << std::endl
        << "<meta charset=\"UTF-8\"/>" << std::endl
        << "<title>"<<prefix<<"</title>" << std::endl
        << "</head>" << std::endl
        << "<body>" << std::endl;
        
    
    state2html.produceTableHeader(site);
    
    
    const unsigned numInstances = dynamic_cast<convnet::TensorData&>(*m_stagingBuffer.outputs[0]).getValues().getNumInstances();
    std::vector<unsigned> allInstances;
    for (unsigned i = 0; i < numInstances; i++)
        allInstances.push_back(i);

    unsigned numSamples = 0;

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

      
        if (numSamples < 32) {
            state2html.addTableRows(m_state, site, report, allInstances);
        } else break;
        numSamples += numInstances;
    }
    state2html.produceTableFooter(site);
    site
        << "</body>" << std::endl
        << "</html>" << std::endl;
        
    report.putText(site, prefix+".html");
}



void WholeImageClassificationTrainer::outputBestWorstN(TrainingDataSource &validationSet, const std::string &prefix, unsigned N)
{
    struct Img {
        RasterImage imageData;
        unsigned GTlabel;
        std::vector<std::pair<float, unsigned>> predictions;
    };
    

    std::vector<Img> images;
    
    images.resize(N);
    
    
    validationSet.restart();
    unsigned numLabels = m_convnet.getConnectionList().outputs[m_outputLabels].size.channels;
    
    unsigned numSamples = 0;
    
    while (true) {
        m_state.clear(*m_computeStream);

        std::cout << "\r Running validation sample " << numSamples << "                      ";

        if (!validationSet.produceMinibatch(m_stagingBuffer)) {
            break;
        }
        {
            TensorData &label = dynamic_cast<TensorData&>(*m_stagingBuffer.outputs[1]);
            TensorData &data = dynamic_cast<TensorData&>(*m_stagingBuffer.outputs[0]);

            const unsigned numInstances = label.getValues().getNumInstances();

            MappedTensor mappedLabel = label.getValues().lock();
            MappedTensor mappedData = data.getValues().lock();

            for (unsigned n = 0; n < numInstances; n++) {
                if (numSamples + n >= images.size()) break;
                
                images[numSamples+n].imageData.resize(32, 32);
                
                for (unsigned y = 0; y < 32; y++)
                    for (unsigned x = 0; x < 32; x++) {
                        int r = std::min(std::max<int>(mappedData.get<float>(x, y, 0, 0, n) * 127.0f + 127.0f, 0), 255);
                        int g = std::min(std::max<int>(mappedData.get<float>(x, y, 0, 1, n) * 127.0f + 127.0f, 0), 255);
                        int b = std::min(std::max<int>(mappedData.get<float>(x, y, 0, 2, n) * 127.0f + 127.0f, 0), 255);
                        
                        images[numSamples+n].imageData.getData()[x+y*32] = 
                                            (r << 0) |
                                            (g << 8) |
                                            (b << 16) |
                                            (0xFF << 24);
                        
                    }                            
            }
        
        
            label.getValues().unlock(mappedLabel);
            data.getValues().unlock(mappedData);
    
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
            AddCudaScopedProfileInterval("Computing label error");

            convnet::TensorData &inputLabel = dynamic_cast<convnet::TensorData&>(*m_state.outputs[m_outputInputLabels]);
            convnet::TensorData &outputLabel = dynamic_cast<convnet::TensorData&>(*m_state.outputs[m_outputLabels]);

            convnet::MappedTensor mappedInputLabel = inputLabel.getValues().lock();
            convnet::MappedTensor mappedOutputLabel = outputLabel.getValues().lock();


            std::vector<float> exps;
            
            for (unsigned n = 0; n < mappedOutputLabel.getNumInstances(); n++) {
                if (numSamples + n >= images.size()) break;

                int label = mappedInputLabel.get<float>(0, 0, 0, 0, n);
                if (label == 0xFF) continue;

                float maxOut = mappedOutputLabel.get<float>(0, 0, 0, 0, n);
                for (unsigned c = 0; c < mappedOutputLabel.getSize().channels; c++)
                    maxOut = std::max(maxOut, mappedOutputLabel.get<float>(0, 0, 0, c, n));

                exps.resize(mappedOutputLabel.getSize().channels);
                float denom = 0.0f;
                for (unsigned c = 0; c < mappedOutputLabel.getSize().channels; c++) {
                    exps[c] = std::exp(mappedOutputLabel.get<float>(0, 0, 0, c, n) - maxOut);
                    denom += exps[c];
                }

                std::vector<std::pair<float, unsigned>> results;
                for (unsigned c = 0; c < mappedOutputLabel.getSize().channels; c++) {
                    results.push_back({1.0f - exps[c] / denom, c});
                }

                std::sort(results.begin(), results.end());

                images[numSamples+n].GTlabel = label;
                images[numSamples+n].predictions = std::move(results);
            }

            inputLabel.getValues().unlock(mappedInputLabel, true);
            outputLabel.getValues().unlock(mappedOutputLabel, true);            
        }
        
        numSamples += dynamic_cast<convnet::TensorData&>(*m_stagingBuffer.outputs[0]).getValues().getNumInstances();        
        if (numSamples >= images.size()) break;        
    }
    
    for (unsigned i = 0; i < images.size(); i++) {
        images[i].imageData.writeToFile((boost::format("%s_%04d.png") % prefix % i).str().c_str());
        
        std::cout << i << std::endl;
        std::cout << "     GT label: " << images[i].GTlabel << std::endl;
        for (unsigned j = 0; j < 3; j++) {
            std::cout << "     " << images[i].predictions[j].second << "  (" << 100-images[i].predictions[j].first*100 << "%)"<<std::endl;
        }        
    }
}


}
