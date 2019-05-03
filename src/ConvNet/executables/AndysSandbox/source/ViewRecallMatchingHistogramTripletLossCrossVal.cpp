/*
 * Mental Image Retrieval - C++ Machine Learning implementation
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

#include "ViewRecallMatchingHistogramTripletLossCrossVal.h"

#include <convnet/Blobb.h>
#include <convnet/Convnet.h>
#include <convnet/layers/InputLayer.h>
#include <convnet/layers/ConvolutionLayer.h>
#include <convnet/layers/BatchNormLayer.h>
#include <convnet/layers/ReluLayer.h>
#include <convnet/layers/PoolingLayer.h>
#include <convnet/layers/DropoutLayer.h>
#include <convnet/layers/CrossEntropyLossLayer.h>
#include <convnet/layers/TripletLossLayer.h>
#include <convnet/layers/BatchNormLayer.h>
#include <convnet/backend/ConvNetImpl_cudnn.h>
#include <convnet/backend/CuDnnWrapper.h>
#include <convnet/backend/TensorImpl_cudnn.h>
#include <convnet/training/Trainer.h>
#include <convnet/reporting/HTTPReport.h>


#include <tools/CPUStopWatch.h>

#include <boost/format.hpp>

#include <random>
#include <vector>
#include <assert.h>
#include <iostream>

class ViewRecallMatchingHistogramTripletLossCrossValTrainer : public convnet::Trainer
{
    public:
        ViewRecallMatchingHistogramTripletLossCrossValTrainer(convnet::ConvNet &convnet, HistogramDataSource &dataSource) : Trainer(convnet, dataSource) { 
            m_lossLayers[0] = convnet.getLayer<convnet::CrossEntropyLossLayer>("loss_layer_1");
            m_lossLayers[1] = convnet.getLayer<convnet::CrossEntropyLossLayer>("loss_layer_2");

            m_viewDescOutput = convnet.getConnectionList().findOutput("desc_view");
            m_recallDescOutput = convnet.getConnectionList().findOutput("desc_recall");

            m_lossOutputs[0] = convnet.getConnectionList().findOutput("loss_1");
            m_lossOutputs[1] = convnet.getConnectionList().findOutput("loss_2");
        }

        void runValidation(HistogramDataSource &dataSource, float &loss) {
            unsigned denom = 0;
            loss = 0.0f;
            runDataset(dataSource, [&](unsigned sampleIndex, convnet::NetworkState &state)->bool {
                loss += sumLoss(m_lossOutputs[0]) + sumLoss(m_lossOutputs[1]);
                denom++;
                return true;
            });
            loss /= denom;
        }

        
        struct SubjectImage {
            unsigned subject, image;
            bool operator<(const SubjectImage &rhs) const {
                if (subject < rhs.subject) return true;
                if (subject > rhs.subject) return false;
                if (image < rhs.image) return true;
                if (image > rhs.image) return false;
                return false;
            }
        };
        struct Match {
            float score;
            unsigned image;
        };
        
        typedef std::vector<Match> Matches;
        typedef std::map<SubjectImage, Matches> RankingResults;

        void runRanking(HistogramDataSource &dataSource, bool subjectBased, RankingResults &results) {

            dataSource.restart();
            
            struct Descriptors {
                std::vector<float> viewDesc;
                std::vector<float> recallDesc;
            };
            
            std::vector<Descriptors> allDescriptors;
            allDescriptors.resize(dataSource.getHistograms().size());
            
            m_computeStream->flush();
            
            const unsigned batchSize = dynamic_cast<convnet::TensorData&>(*m_stagingBuffer.outputs[0]).getValues().getNumInstances();

            unsigned numSamples = 0;
            std::cout << std::endl;
            while (numSamples < dataSource.getHistograms().size()) {
                m_state.clear(*m_computeStream);

                std::cout << "\r Running batch " << numSamples << " of " << dataSource.getHistograms().size() << "                      ";

                for (unsigned i = 0; i < batchSize; i++) {
                    if (numSamples+i >= dataSource.getHistograms().size()) break;
                    
                    const HistogramData::Histogram &hist = dataSource.getHistograms()[numSamples+i];
                    
                    const unsigned viewScanPathOutput = 0;
                    const unsigned recallScanPathOutput = 1;
                    
                    dataSource.uploadHistogram(
                        m_stagingBuffer, 
                        (const char*)&hist.view, 
                        sizeof(HistogramData::Histogram), 
                        viewScanPathOutput, 
                        1, i
                    );
                    
                    dataSource.uploadHistogram(
                        m_stagingBuffer, 
                        (const char*)&hist.recall, 
                        sizeof(HistogramData::Histogram), 
                        recallScanPathOutput, 
                        1, i
                    );
                }

                m_inputLayer->swapInStagingBuffer(m_state, m_stagingBuffer);
                {
                    m_convnet.feedForward(m_state, *m_executionWorkspace, *m_computeStream, false);

                    m_computeStream->flush();
                }

                {
                    auto &descOutput = dynamic_cast<convnet::TensorData&>(*m_state.outputs[m_viewDescOutput]);
                
                    convnet::MappedTensor mapped = descOutput.getValues().lock();
                    for (unsigned n = 0; n < mapped.getNumInstances(); n++) {
                        if (numSamples+n >= dataSource.getHistograms().size()) break;

                        allDescriptors[numSamples+n].viewDesc.resize(mapped.getSize().numElements());
                        memcpy(allDescriptors[numSamples+n].viewDesc.data(), mapped.data<float>(n), mapped.getSize().numElements() * sizeof(float));
                    }
                    descOutput.getValues().unlock(mapped, true);            
                }
                {
                    auto &descOutput = dynamic_cast<convnet::TensorData&>(*m_state.outputs[m_recallDescOutput]);
                
                    convnet::MappedTensor mapped = descOutput.getValues().lock();
                    for (unsigned n = 0; n < mapped.getNumInstances(); n++) {
                        if (numSamples+n >= dataSource.getHistograms().size()) break;

                        allDescriptors[numSamples+n].recallDesc.resize(mapped.getSize().numElements());
                        memcpy(allDescriptors[numSamples+n].recallDesc.data(), mapped.data<float>(n), mapped.getSize().numElements() * sizeof(float));
                    }
                    descOutput.getValues().unlock(mapped, true);            
                }
                
                numSamples += batchSize;
            }
            std::cout << std::endl;
            
            for (unsigned queryIdx = 0; queryIdx < dataSource.getHistograms().size(); queryIdx++) {
                std::cout << "Matching " << queryIdx << " of " << dataSource.getHistograms().size() << std::endl;
/*
                std::cout << "  ";
                for (unsigned i = 0; i < allDescriptors[queryIdx].recallDesc.size(); i++) 
                    std::cout << " " << allDescriptors[queryIdx].recallDesc[i];
                std::cout << std::endl;

                std::cout << "  ";
                for (unsigned i = 0; i < allDescriptors[queryIdx].viewDesc.size(); i++) 
                    std::cout << " " << allDescriptors[queryIdx].viewDesc[i];
                std::cout << std::endl;
*/

                for (unsigned matchIdx = 0; matchIdx < dataSource.getHistograms().size(); matchIdx++) {
                    if (subjectBased && (dataSource.getHistograms()[queryIdx].subject != dataSource.getHistograms()[matchIdx].subject)) continue;
                   
                    float difference = 0.0f;
                    
                    for (unsigned i = 0; i < allDescriptors[queryIdx].recallDesc.size(); i++) {
                        float r = allDescriptors[queryIdx].recallDesc[i];
                        float v = allDescriptors[matchIdx].viewDesc[i];
                        difference += (r-v)*(r-v);
                    }

                    //std::cout << dataSource.getHistograms()[queryIdx].image << "  " << dataSource.getHistograms()[matchIdx].image << "   " << difference << std::endl;
                    
                    results[SubjectImage{dataSource.getHistograms()[queryIdx].subject, dataSource.getHistograms()[queryIdx].image}].push_back(Match{difference, dataSource.getHistograms()[matchIdx].image});
                }
            }
            for (auto &query : results) {
                std::sort(query.second.begin(), query.second.end(), [](const Match &lhs, const Match &rhs)->bool{
                    return lhs.score < rhs.score;
                });
            }
        }
        
        void resetRunningLoss() {
            m_runningLoss = 0.0f;
        }
        
        float getRunningLoss() const { return m_runningLoss; }
    protected:
        convnet::CrossEntropyLossLayer *m_lossLayers[2];
        unsigned m_lossOutputs[2];
        unsigned m_viewDescOutput;
        unsigned m_recallDescOutput;

        float m_runningLoss;
        unsigned runningLossDenom;

        virtual void handleLearningLoss() {
            m_runningLoss += sumLoss(m_lossOutputs[0]) + sumLoss(m_lossOutputs[1]);
        }

};


void trainViewRecallMatchingHistogramTripletLossCrossVal(const eyeTracking::Dataset &dataset, HistogramData dataTrain, HistogramData dataTest, convnet::CuDnnWrapper &cudnnWrapper)
{
    std::mt19937 rng;
    convnet::HTTPReport httpReport("/var/www/reporting/", "../data/reporting/", false);

    
    struct SubjectResults {
		std::vector<float> accuracyVsRank;
    	unsigned accuracyVsRankDenom = 0;
		
		SubjectResults() {
		    accuracyVsRank.resize(10, 0.0f);
		}
    };

    std::vector<SubjectResults> subjectResults;
    subjectResults.resize(dataTest.subjects.size());

    Eigen::MatrixXf totalConfusion;
    totalConfusion.resize(100, 100);
    totalConfusion.setZero();
    
    Eigen::MatrixXf totalConfusion_top3;
    totalConfusion_top3.resize(100, 100);
    totalConfusion_top3.setZero();

    std::vector<float> averageAccuracyVsRank;
    averageAccuracyVsRank.resize(10, 0.0f);
    unsigned averageAccuracyVsRankDenom = 0;


    auto printConfusionMatrix = [&](std::stringstream &html, const Eigen::MatrixXf &matrix) {
        html << "<table border=\"1px\">" << std::endl;
        for (unsigned i = 0; i < matrix.rows(); i++) {
            html << "<tr>" << std::endl;
            for (unsigned j = 0; j < matrix.cols(); j++)
                html << "<td><span title=\"" << dataTest.images[i].filename << " classified as " << dataTest.images[j].filename << "\">" << matrix(i, j) << "</span></td>" << std::endl;            
            html << "</tr>" << std::endl;
        }
        html << "</table>" << std::endl;
        html << "<font size=\"2\"><pre>[";
        for (unsigned i = 0; i < matrix.rows(); i++) {
            if (i > 0)
                html << ", ";
            html << '[';
            for (unsigned j = 0; j < matrix.cols(); j++) {
                if (j > 0)
                    html << ", ";
                html << matrix(i, j);
            }
            html << ']';
        }
        html << "]</pre></font> " << std::endl;
    };

    auto printAccuracyVsRank = [&](std::stringstream &html, const std::vector<float> &accuracyVsRank) {
        html << "<font size=\"2\"><pre>[";
        for (unsigned i = 0; i < accuracyVsRank.size(); i++) {
            if (i > 0)
                html << ", ";
            html << accuracyVsRank[i];
        }
        html << "]</pre></font> " << std::endl;
    };

    auto printAccuracyVsRankSubjects = [&](std::stringstream &html, const std::vector<SubjectResults> &subjectResults) {
        html << "<font size=\"2\"><pre>[";
		for (unsigned j = 0; j < subjectResults.size(); j++) {
			if (j > 0)
				html << ", [";
			else
				html << "[";

		    for (unsigned i = 0; i < subjectResults[j].accuracyVsRank.size(); i++) {
		        if (i > 0)
		            html << ", ";
		        html << subjectResults[j].accuracyVsRank[i] / subjectResults[j].accuracyVsRankDenom;
		    }
			html << "]";
		}
        html << "]</pre></font> " << std::endl;
    };
    
    auto updateResults = [&]() {
        std::stringstream main;
        main
            << "<!DOCTYPE html>" << std::endl
            << "<html>" << std::endl
            << "<head>" << std::endl
            << "<meta charset=\"UTF-8\"/>" << std::endl
            << "<title>Eye movement</title>" << std::endl
            << "</head>" << std::endl

            << "<body>" << std::endl;

        main << "<h1>Average over " << averageAccuracyVsRankDenom << " images</h1>" << std::endl;

        std::vector<float> avg = averageAccuracyVsRank;
        for (unsigned i = 0; i < avg.size(); i++)
            avg[i] /= averageAccuracyVsRankDenom;
        
        float AUC = 0.0f;
        for (unsigned i = 0; i < avg.size(); i++)
            AUC += avg[i];
        AUC /= avg.size();
        
        main << "AUC: " << AUC << "<br/>" << std::endl;


        main << "accuracyVsRank: ";
        printAccuracyVsRank(main, avg);
        
        main << "<h1>Total confusion</h1>" << std::endl;
        printConfusionMatrix(main, totalConfusion);
        main << "<h1>Total confusion top3</h1>" << std::endl;
        printConfusionMatrix(main, totalConfusion_top3);

        main << "<h1>Per subject</h1>" << std::endl;

        main << "Accuracy vs rank split by subjects:" << std::endl;
		printAccuracyVsRankSubjects(main, subjectResults);
   
        main << "<font size=\"2\"><pre>subjects = [";
        for (unsigned s = 0; s < subjectResults.size(); s++) {
            if (s > 0)
                main << ", ";
            main << "'" << dataTest.subjects[s].name << "'";
        }
        main << "]</pre></font> " << std::endl;  

        main << "<h1>Image list</h1>" << std::endl;
        for (unsigned i = 0; i < dataTest.images.size(); i++) {
            main << "image "<<i<<": " << dataTest.images[i].filename << "<br/>" << std::endl;
        }
        main << "<font size=\"2\"><pre>images = [";
        for (unsigned i = 0; i < dataTest.images.size(); i++) {
            if (i > 0)
                main << ", ";
            main << "'" << dataTest.images[i].filename << "'";
        }
        main << "]</pre></font> " << std::endl;          
        
        main
            << "</body>" << std::endl
            << "</html>" << std::endl;
        httpReport.putText(main, "matchingViewRecall_2.html");
    };
    
    updateResults();
    
    for (unsigned s = 0; s < dataTest.subjects.size(); s++) {
        if (dataTest.subjects[s].name != dataTrain.subjects[s].name) 
            throw std::runtime_error("train and test mismatch on subjects!");
    }

    for (unsigned s = 0; s < dataTest.images.size(); s++) {
        if (dataTest.images[s].filename != dataTrain.images[s].filename) 
            throw std::runtime_error("train and test mismatch on images!");
    }
    
    
    for (unsigned fold = 0; fold < 10; fold++) {
    convnet::ConvNetImpl_cudnn net(cudnnWrapper);
        convnet::TensorSize inputSize(24, 24, 1, 1);

        const float wd = 0.001f;
        net.addLayer<convnet::InputLayer>("input")->setOutputConnectionNames({
            "view", "recall",
            "opposing_view", "opposing_recall"
        });
        
        
        auto buildNetworkBranch = [&](const std::string &prefix, const std::string &input, const std::string &output) {

            bool useBatchNorm = true;

    unsigned layerScale = 4;

            net.addLayer<convnet::ConvolutionLayer>("B_C1", prefix+"_C_1")->seedParametersGaussianXavier(layerScale, convnet::TensorSize(3, 3, 1, 1), rng, true, 1.0f, false)
                            ->setWeightDecay(useBatchNorm?0.0f:1e-8f)
                            ->setHasBias(!useBatchNorm)
                            ->setInputConnectionNames({input});
            net.addLayer<convnet::ReluLayer>();
            net.addLayer<convnet::ConvolutionLayer>("B_C2", prefix+"_C_2")->seedParametersGaussianXavier(layerScale*2, convnet::TensorSize(3, 3, 1, layerScale), rng)
                            ->setWeightDecay(useBatchNorm?0.0f:1e-8f)
                            ->setHasBias(!useBatchNorm);
            net.addLayer<convnet::ReluLayer>();

            net.addLayer<convnet::DropoutLayer>()
                ->setDropout(0.3f)
                ->setSeed(rng());    

            net.addLayer<convnet::PoolingLayer>()->setWindowSize(2, 2, 1)->setMode(convnet::PoolingLayer::MODE_MAX);

            net.addLayer<convnet::ConvolutionLayer>("B_C3", prefix+"_C_3")->seedParametersGaussianXavier(layerScale*2, convnet::TensorSize(3, 3, 1, layerScale*2), rng)
                            ->setWeightDecay(useBatchNorm?0.0f:1e-8f)
                            ->setHasBias(!useBatchNorm);
            net.addLayer<convnet::ReluLayer>();
            net.addLayer<convnet::ConvolutionLayer>("B_C4", prefix+"_C_4")->seedParametersGaussianXavier(layerScale*4, convnet::TensorSize(3, 3, 1, layerScale*2), rng)
                            ->setWeightDecay(useBatchNorm?0.0f:1e-8f)
                            ->setHasBias(!useBatchNorm);
            net.addLayer<convnet::ReluLayer>();

            net.addLayer<convnet::DropoutLayer>()
                ->setDropout(0.3f)
                ->setSeed(rng());    
            
            net.addLayer<convnet::PoolingLayer>()->setWindowSize(2, 2, 1)->setMode(convnet::PoolingLayer::MODE_MAX);
            
            unsigned w = ((inputSize.width - 4)/2 - 4)/2;
            unsigned c = 256;//w*w*128 / 4;
            net.addLayer<convnet::ConvolutionLayer>("B_C5", prefix+"_C_5")->seedParametersGaussianXavier(c, convnet::TensorSize(w, w, 1, layerScale*4), rng, true, 1.0f, false)
                            ->setWeightDecay(useBatchNorm?0.0f:1e-8f)
                            ->setHasBias(!useBatchNorm);
            net.addLayer<convnet::ReluLayer>();

            net.addLayer<convnet::DropoutLayer>()
                ->setDropout(0.2f)
                ->setSeed(rng());    
            net.addLayer<convnet::ConvolutionLayer>("B_C6", prefix+"_C_6")->seedParametersGaussianXavier(16, convnet::TensorSize(1, 1, 1, c), rng)
                        ->setWeightDecay(1e-8f)
                        ->setOutputConnectionNames({output});
        };



        buildNetworkBranch(
            "d_view",
            "view",
            "desc_view"
        );

        buildNetworkBranch(
    //        "d_view",
            "d_recall",
            "recall",
            "desc_recall"
        );

        buildNetworkBranch(
            "d_view",
            "opposing_view",
            "desc_opposing_view"
        );

        buildNetworkBranch(
    //        "d_view",
            "d_recall",
            "opposing_recall",
            "desc_opposing_recall"
        );

        
        net.addLayer<convnet::TripletLossLayer>("loss_layer_1")
            ->setInputConnectionNames({"desc_view", "desc_recall", "desc_opposing_recall"})
            ->setOutputConnectionNames({"loss_1"});

        net.addLayer<convnet::TripletLossLayer>("loss_layer_2")
            ->setInputConnectionNames({"desc_recall", "desc_view", "desc_opposing_view"})
            ->setOutputConnectionNames({"loss_2"});
            

        net.saveToFile("InitialNet.xml");

        net.connectLayers();

        
        std::set<unsigned> testFold;
        for (unsigned i = 0; i < 10; i++)
            testFold.insert(fold * 10 + i);
        
        
        HistogramData dataCopyTrain = dataTrain;
        HistogramData dataCopyTest = dataTest;
        
        
        HistogramDataSource testDataSource;
        testDataSource.takeImages(dataCopyTest, testFold);
        {
            HistogramDataSource dummy;
            dummy.takeImages(dataCopyTrain, testFold);
        }

        HistogramDataSource trainingDataSource;
        trainingDataSource.takeAll(dataCopyTrain);

        bool subjectBased = true;

        auto setDataSourceOutputs = [subjectBased](HistogramDataSource &dataSource) {
            dataSource.viewOutput = 0;
            dataSource.recallOutput = 1;
            dataSource.opposingViewOutput = 2;
            dataSource.opposingRecallOutput = 3;
            
            dataSource.opposingSequenceFromSameSubject = subjectBased;
        };
        setDataSourceOutputs(testDataSource);
        setDataSourceOutputs(trainingDataSource);




        convnet::InputLayer *inputLayer = net.getLayer<convnet::InputLayer>("input");
        inputLayer->setInputSizes({
            inputSize,
            inputSize,
            inputSize,
            inputSize
        });

        auto computeStream = net.allocateExecutionStream();

        const unsigned batchSize = 20;
        net.resizeLayers(*computeStream, batchSize);

        ViewRecallMatchingHistogramTripletLossCrossValTrainer trainer(net, trainingDataSource);
        
        net.pushParametersToBackend(trainer.getExecutionWorkspace(), trainer.getComputeStream());

        std::vector<std::pair<unsigned, float>> learningLossMatch;
        std::vector<std::pair<unsigned, float>> learningLossNonMatch;
        std::vector<std::pair<unsigned, float>> testLoss;
        std::vector<std::pair<unsigned, float>> testTP;
        std::vector<std::pair<unsigned, float>> testTN;
        
        std::vector<std::pair<unsigned, float>> testAccuracy;
        std::vector<float> lastTestROC;
        std::vector<float> lastTrainROC;
        std::vector<std::pair<unsigned, float>> trainingErrorAUC;
        std::vector<std::pair<unsigned, float>> testErrorAUC;
        
        unsigned burninIters = 100;
        unsigned currentBurninIter = 0;

        float learningRate = -0.001f;
        
        
        auto updateReport = [&](unsigned epoch, unsigned iter) {
            std::stringstream main;
            main
                << "<!DOCTYPE html>" << std::endl
                << "<html>" << std::endl
                << "<head>" << std::endl
                << "<meta charset=\"UTF-8\"/>" << std::endl
                << "<title>Eye movement</title>" << std::endl
                << "<script src=\"js/plotly-latest.min.js\"></script>" << std::endl
                << "</head>" << std::endl

                << "<body>" << std::endl
                    << "epoch " << epoch << " iter " << iter << std::endl
                    << "<div id=\"chartDiv1\"></div><br/>" << std::endl
                    << "<div id=\"chartDiv2\"></div>" << std::endl
                    << "<script>" << std::endl;
            
            auto putArrayXY = [](std::ostream &stream, const std::string &varName, const std::string &name, bool secondAxis, const std::vector<std::pair<unsigned, float>> &array) {
                stream << "var " << varName << " = { ";
                stream << " x: [";
                for (unsigned i = 0; i < array.size(); i++) 
                    if (i > 0)
                        stream << ", " << array[i].first;
                    else
                        stream << ' ' << array[i].first;
                stream << "],";

                stream << " y: [";
                for (unsigned i = 0; i < array.size(); i++) 
                    if (i > 0)
                        stream << ", " << array[i].second;
                    else
                        stream << ' ' << array[i].second;
                stream << "],";
                
                stream << " name: \""<<name<<"\", ";

                if (secondAxis)
                    stream << "yaxis: 'y2', ";

                stream << "type: 'scatter'};" << std::endl;
            };
            auto putArray = [](std::ostream &stream, const std::string &varName, const std::string &name, bool secondAxis, const std::vector<float> &array) {
                stream << "var " << varName << " = { ";
                stream << " y: [";
                for (unsigned i = 0; i < array.size(); i++) 
                    if (i > 0)
                        stream << ", " << array[i];
                    else
                        stream << ' ' << array[i];
                stream << "],";
                
                stream << " name: \""<<name<<"\", ";

                if (secondAxis)
                    stream << "yaxis: 'y2', ";

                stream << "type: 'scatter'};" << std::endl;
            };
        
            putArrayXY(main, "learningLossMatch", "training error", false, learningLossMatch);
            putArrayXY(main, "testLoss", "test loss", false, testLoss);
            putArrayXY(main, "testAccuracy", "test accuracy", true, testAccuracy);
            putArrayXY(main, "trainingErrorAUC", "training AUC", true, trainingErrorAUC);
            putArrayXY(main, "testErrorAUC", "test AUC", true, testErrorAUC);


            main
                    << "var data = [learningLossMatch, testLoss, testAccuracy, trainingErrorAUC, testErrorAUC];" << std::endl
                    << "var layout = {title: 'Losses over time', xaxis: {title: 'epoch'}, yaxis: {title: 'X entropy loss'}, yaxis2: { title: 'Accuracy', overlaying: 'y', side: 'right'}};" << std::endl
                    << "Plotly.newPlot('chartDiv1', data, layout);" << std::endl;
                    
            putArray(main, "lastTestROC", "test roc", false, lastTestROC);
            putArray(main, "lastTrainROC", "train roc", false, lastTrainROC);
            main
                    << "var data2 = [lastTestROC, lastTrainROC];" << std::endl
                    << "var layout2 = {title: 'ROC', yaxis: {title: 'Accuracy'}, xaxis: {title: 'Rank'}};" << std::endl
                    << "Plotly.newPlot('chartDiv2', data2, layout2);" << std::endl;
                    
            main
                    << "</script>" << std::endl
                << "</body>" << std::endl
                << "</html>" << std::endl;
            httpReport.putText(main, "index_eye_normalized_2.html");
        };
        
        
        
        for (unsigned epoch = 0; epoch < 1000; epoch++) {
            trainingDataSource.restart();
            trainer.resetRunningLoss();
            unsigned iter = 0;        
            
            while (true) {
                std::cout << "epoch " << epoch << std::endl;
                std::cout << "        iter " << iter << std::endl;
                std::cout << "            learningRate " << learningRate << std::endl;
            //   if (iter % 1000 == 0)
                //    plot((boost::format("adv_test_%08d")%iter).str());

                if (currentBurninIter < burninIters) {
                    currentBurninIter++;
                    std::cout << "Burnin lr " << learningRate * std::exp(((float)currentBurninIter - (float)burninIters) * 0.1f) << std::endl;
                    if (!trainer.learn(1, learningRate * std::exp(((float)currentBurninIter - (float)burninIters) * 0.1f))) break;
                } else
                    if (!trainer.learn(1, learningRate)) break;

                iter++;
            }

    #if 1
            if (epoch % 100 == 0) {
                std::cout << "Running validation" << std::endl;
                Engine::CPUStopWatch stopWatch;

                float loss;
                trainer.runValidation(testDataSource, loss);
                testLoss.push_back({epoch, loss / batchSize});
                
                learningLossMatch.push_back({epoch, trainer.getRunningLoss() / (iter * batchSize)});
                
                ViewRecallMatchingHistogramTripletLossCrossValTrainer::RankingResults results;
                trainer.runRanking(testDataSource, subjectBased, results);
                {
                    unsigned numCorrect = 0;
                    for (const auto &result : results) {
                        if (result.first.image == result.second.front().image)
                            numCorrect++;
                    }
                    testAccuracy.push_back({epoch, numCorrect / (float) results.size()});
                }
                {
                    lastTestROC.clear();
                    lastTestROC.resize(results.begin()->second.size(), 0);
                    
                    for (const auto &result : results) {
                        for (unsigned j = 0; j < lastTestROC.size(); j++) {
                            if (result.second[j].image == result.first.image) {
                                for (unsigned k = j; k < lastTestROC.size(); k++) {
                                    lastTestROC[k] += 1.0f / results.size();
                                }
                                break;
                            }
                        }
                    }
                    
                    {
                        float AUC = 0.0f;
                        for (float f : lastTestROC)
                            AUC += f;
                        AUC /= lastTestROC.size();
                        testErrorAUC.push_back({epoch, AUC});
                    }                
                }
    #if 1
                if (epoch % 100 == 0) {
                    ViewRecallMatchingHistogramTripletLossCrossValTrainer::RankingResults resultsTrain;
                    trainer.runRanking(trainingDataSource, subjectBased, resultsTrain);
                    {
                        lastTrainROC.clear();
                        lastTrainROC.resize(resultsTrain.begin()->second.size(), 0);
                        
                        for (const auto &result : resultsTrain) {
                            for (unsigned j = 0; j < lastTrainROC.size(); j++) {
                                if (result.second[j].image == result.first.image) {
                                    for (unsigned k = j; k < lastTrainROC.size(); k++) {
                                        lastTrainROC[k] += 1.0f / resultsTrain.size();
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    {
                        float AUC = 0.0f;
                        for (float f : lastTrainROC)
                            AUC += f;
                        AUC /= lastTrainROC.size();
                        trainingErrorAUC.push_back({epoch, AUC});
                    }
                }
    #endif
                updateReport(epoch+1, 0);

                std::cout << "Running validation took " << stopWatch.getNanoseconds() * 1e-9f << " seconds!" << std::endl;
            }
            //trainer.debugOutputTest(validationDataSource, "VAE_CIFAR_", 128);
    //        trainer.debugOutputTest(trainingDataSource, "gan_transcoding2_", 32);
            //if (epoch % 1 == 0)
            //  trainer.report(trainingDataSource, httpReport, "train", rendering_inputChannelSelection, rendering_outputChannelSelection);


    #endif
            if (epoch % 100 == 99)
                learningRate *= 0.75f;
        }
        
        {
            std::cout << "Running validation" << std::endl;
            Engine::CPUStopWatch stopWatch;

            ViewRecallMatchingHistogramTripletLossCrossValTrainer::RankingResults results;
            trainer.runRanking(testDataSource, subjectBased, results);
            
            for (const auto &result : results) {
                totalConfusion(result.first.image, result.second[0].image) += 1.0f;
                for (unsigned k = 0; k < 3; k++)
                    totalConfusion_top3(result.first.image, result.second[k].image) += 1.0f;
            }            

            for (const auto &result : results) {
                for (unsigned j = 0; j < averageAccuracyVsRank.size(); j++) {
                    if (result.second[j].image == result.first.image) {
                        for (unsigned k = j; k < averageAccuracyVsRank.size(); k++) {
                            averageAccuracyVsRank[k] += 1.0f / results.size();
							subjectResults[result.first.subject].accuracyVsRank[k] += 1.0f;
                        }
                        break;
                    }
                }
				subjectResults[result.first.subject].accuracyVsRankDenom++;
            }
            averageAccuracyVsRankDenom++;
        }
	    updateResults();
    }
}
