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

#include "ImageClassificationMatrixPlots.h"
#include <convnet/Convnet.h>
#include <convnet/layers/InputLayer.h>
#include <convnet/layers/ConvolutionLayer.h>
#include <convnet/layers/ReluLayer.h>
#include <convnet/layers/PoolingLayer.h>
#include <convnet/layers/DropoutLayer.h>
#include <convnet/layers/BatchNormLayer.h>
#include <convnet/layers/CrossEntropyLossLayer.h>
#include <convnet/backend/ConvNetImpl_cudnn.h>
#include <convnet/backend/CuDnnWrapper.h>
#include <convnet/backend/TensorImpl_cudnn.h>
#include <convnet/training/WholeImageClassificationTrainer.h>
#include <convnet/reporting/HTTPReport.h>

#include <tools/CPUStopWatch.h>

#include <boost/format.hpp>

#include <random>
#include <vector>
#include <assert.h>
#include <iostream>


void imageClassificationmatrixPlots(eyeTracking::Dataset &dataset, HistogramData dataTrain, HistogramData dataTest, convnet::CuDnnWrapper &cudnnWrapper)
{
    convnet::HTTPReport httpReport("/var/www/reporting/", "../data/reporting/", false);
    
    std::mt19937 rng;
    
    struct SubjectResults {
        float top1;
        float top3;
        float AUC;

		std::vector<float> accuracyVsRank;
        
        Eigen::MatrixXf confusion;
        Eigen::MatrixXf confusion_top3;

        
        SubjectResults() {
            top1 = top3 = AUC = 0.0f;
            confusion.resize(100, 100);
            confusion.setZero();            
            confusion_top3.resize(100, 100);
            confusion_top3.setZero();            
        }
    };

	float averageAUC = 0.0f;
	unsigned averageDenom = 0;
    
    Eigen::MatrixXf totalConfusion;
    totalConfusion.resize(100, 100);
    totalConfusion.setZero();
    
    Eigen::MatrixXf totalConfusion_top3;
    totalConfusion_top3.resize(100, 100);
    totalConfusion_top3.setZero();

    std::vector<SubjectResults> subjectResults;
    subjectResults.resize(dataTest.subjects.size());


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

        main << "<h1>Average over " << averageDenom << " subjects</h1>" << std::endl;

        main << "AUC: " << averageAUC / averageDenom << "<br/>" << std::endl;

		std::vector<float> avg;
		avg.resize(subjectResults.front().accuracyVsRank.size(), 0.0f);
        for (unsigned s = 0; s < averageDenom; s++)
			for (unsigned i = 0; i < avg.size(); i++)
				avg[i] += subjectResults[s].accuracyVsRank[i];
		for (unsigned i = 0; i < avg.size(); i++)
			avg[i] /= averageDenom;

		main << "accuracyVsRank: ";
		printAccuracyVsRank(main, avg);

        main << "<h1>Subject performance (Overview)</h1>" << std::endl;
        for (unsigned s = 0; s < subjectResults.size(); s++) {
            main << "<h2>" << dataTest.subjects[s].name << "</h2>" << std::endl;
            main << "top1: " << subjectResults[s].top1 << "<br/>" << std::endl;
            main << "top3: " << subjectResults[s].top3 << "<br/>" << std::endl;
            main << "AUC: " << subjectResults[s].AUC << "<br/>" << std::endl;
        }
        
        main << "<font size=\"2\"><pre>subjects = [";
        for (unsigned s = 0; s < subjectResults.size(); s++) {
            if (s > 0)
                main << ", ";
            main << "'" << dataTest.subjects[s].name << "'";
        }
        main << "]</pre></font> " << std::endl;        

        main << "<font size=\"2\"><pre>top_1 = [";
        for (unsigned s = 0; s < subjectResults.size(); s++) {
            if (s > 0)
                main << ", ";
            main << subjectResults[s].top1;
        }
        main << "]</pre></font> " << std::endl;        
        main << "<font size=\"2\"><pre>top_3 = [";
        for (unsigned s = 0; s < subjectResults.size(); s++) {
            if (s > 0)
                main << ", ";
            main << subjectResults[s].top3;
        }
        main << "]</pre></font> " << std::endl;        
        main << "<font size=\"2\"><pre>AUC = [";
        for (unsigned s = 0; s < subjectResults.size(); s++) {
            if (s > 0)
                main << ", ";
            main << subjectResults[s].AUC;
        }
        main << "]</pre></font> " << std::endl;        

        main << "<h1>Total confusion</h1>" << std::endl;
        printConfusionMatrix(main, totalConfusion);
        main << "<h1>Total confusion top3</h1>" << std::endl;
        printConfusionMatrix(main, totalConfusion_top3);

   
        main << "<h1>Subject performance</h1>" << std::endl;
        for (unsigned s = 0; s < subjectResults.size(); s++) {
            main << "<h2>" << dataTest.subjects[s].name << "</h2>" << std::endl;
            main << "top1: " << subjectResults[s].top1 << "<br/>" << std::endl;
            main << "top3: " << subjectResults[s].top3 << "<br/>" << std::endl;
            main << "AUC: " << subjectResults[s].AUC << "<br/>" << std::endl;
            
            main << "Image confusion matrix:<br/>" << std::endl;
            printConfusionMatrix(main, subjectResults[s].confusion);

            main << "accuracyVsRank:<br/>" << std::endl;
			printAccuracyVsRank(main, subjectResults[s].accuracyVsRank);
        }            

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
        httpReport.putText(main, "matrixPlotsPreTrained.html");
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

    std::set<unsigned> ignoredSubjects;
#if 0
    for (unsigned s = 0; s < dataTest.subjects.size(); s++) {
        if (
                (dataTest.subjects[s].name == "jg") ||
                (dataTest.subjects[s].name == "cg") ||
                (dataTest.subjects[s].name == "gr") ||
                (dataTest.subjects[s].name == "mr") ||
                (dataTest.subjects[s].name == "lg") ||
                (dataTest.subjects[s].name == "ik") ||
                (dataTest.subjects[s].name == "ac") ||
                (dataTest.subjects[s].name == "me") ||
                (dataTest.subjects[s].name == "aw") ||
                (dataTest.subjects[s].name == "dg") ||
                (dataTest.subjects[s].name == "tt") ||
                (dataTest.subjects[s].name == "tp") ||
                (dataTest.subjects[s].name == "mg") ||
                (dataTest.subjects[s].name == "dk") ||
                (dataTest.subjects[s].name == "eb") ||
                (dataTest.subjects[s].name == "pe")
            )
            ignoredSubjects.insert(s);
    }
#endif
    
    for (unsigned droppedSubject = 0; droppedSubject < dataTest.subjects.size(); droppedSubject++) {
        
        convnet::ConvNetImpl_cudnn net(cudnnWrapper);

        convnet::TensorSize inputSize(24, 24, 1, 1);

		bool useBatchNorm = true;

unsigned layerScale = 4;

        net.addLayer<convnet::InputLayer>("input")->setOutputConnectionNames({"data", "label"});
        net.addLayer<convnet::ConvolutionLayer>()->seedParametersGaussianXavier(layerScale, convnet::TensorSize(3, 3, 1, 1), rng, true, 1.0f, false)
                        ->setWeightDecay(useBatchNorm?0.0f:1e-8f)
						->setHasBias(!useBatchNorm)
                        ->setInputConnectionNames({"data"});
		net.addLayer<convnet::BatchNormLayer>()
					->initialize(layerScale, rng);
        net.addLayer<convnet::ReluLayer>();
        net.addLayer<convnet::ConvolutionLayer>()->seedParametersGaussianXavier(layerScale*2, convnet::TensorSize(3, 3, 1, layerScale), rng)
                        ->setWeightDecay(useBatchNorm?0.0f:1e-8f)
						->setHasBias(!useBatchNorm);
		net.addLayer<convnet::BatchNormLayer>()
					->initialize(layerScale*2, rng);
        net.addLayer<convnet::ReluLayer>();

        net.addLayer<convnet::DropoutLayer>()
            ->setDropout(0.3f)
            ->setSeed(rng());    

        net.addLayer<convnet::PoolingLayer>()->setWindowSize(2, 2, 1)->setMode(convnet::PoolingLayer::MODE_MAX);

        net.addLayer<convnet::ConvolutionLayer>()->seedParametersGaussianXavier(layerScale*2, convnet::TensorSize(3, 3, 1, layerScale*2), rng)
                        ->setWeightDecay(useBatchNorm?0.0f:1e-8f)
						->setHasBias(!useBatchNorm);
		net.addLayer<convnet::BatchNormLayer>()
					->initialize(layerScale*2, rng);
        net.addLayer<convnet::ReluLayer>();
        net.addLayer<convnet::ConvolutionLayer>()->seedParametersGaussianXavier(layerScale*4, convnet::TensorSize(3, 3, 1, layerScale*2), rng)
                        ->setWeightDecay(useBatchNorm?0.0f:1e-8f)
						->setHasBias(!useBatchNorm);
		net.addLayer<convnet::BatchNormLayer>()
					->initialize(layerScale*4, rng);
        net.addLayer<convnet::ReluLayer>();

        net.addLayer<convnet::DropoutLayer>()
            ->setDropout(0.3f)
            ->setSeed(rng());    
        
        net.addLayer<convnet::PoolingLayer>()->setWindowSize(2, 2, 1)->setMode(convnet::PoolingLayer::MODE_MAX);
        
        unsigned w = ((inputSize.width - 4)/2 - 4)/2;
        unsigned c = 256;//w*w*128 / 4;
        net.addLayer<convnet::ConvolutionLayer>()->seedParametersGaussianXavier(c, convnet::TensorSize(w, w, 1, layerScale*4), rng, true, 1.0f, false)
                        ->setWeightDecay(useBatchNorm?0.0f:1e-8f)
						->setHasBias(!useBatchNorm);
		net.addLayer<convnet::BatchNormLayer>()
					->initialize(c, rng);
        net.addLayer<convnet::ReluLayer>();

        net.addLayer<convnet::DropoutLayer>()
            ->setDropout(0.2f)
            ->setSeed(rng());    
        net.addLayer<convnet::ConvolutionLayer>()->seedParametersGaussianXavier(100, convnet::TensorSize(1, 1, 1, c), rng)
                    ->setWeightDecay(1e-8f)
                    ->setOutputConnectionNames({"inferredLabel"});

        net.addLayer<convnet::CrossEntropyLossLayer>()
                ->setInputConnectionNames({"inferredLabel", "label"})
                ->setOutputConnectionNames({"loss"});



        net.saveToFile("InitialNet.xml");
        
        

    for (unsigned i = 0; i < net.getNumLayer(); i++) {
        convnet::Layer *layer = net.getLayer(i);
        convnet::ConvolutionLayer *convLayer = dynamic_cast<convnet::ConvolutionLayer *>(layer);
        if (convLayer != nullptr) {
            //convLayer->seedParametersGaussianXavier(convLayer->getParameters()->m_kernel.size(), convLayer->getParameters()->m_filterSize, rng, true, 1.0f, false);
//            convLayer->seedParametersGaussian(convLayer->getParameters()->m_kernel.size(), convLayer->getParameters()->m_filterSize, rng);
        }
    }        

        net.connectLayers();

        
        HistogramData dataCopyTrain = dataTrain;
        HistogramData dataCopyTest = dataTest;
        
        HistogramDataSource testDataSource;
        testDataSource.takeSubjects(dataCopyTest, {droppedSubject});

		{
        	HistogramDataSource dummySource;
	        dummySource.takeSubjects(dataCopyTrain, {droppedSubject});
		}

        HistogramDataSource trainingDataSource;
        trainingDataSource.takeAll(dataCopyTrain);
        //trainingDataSource.takeSubjects(dataCopy, ignoredSubjects);
        
        auto setDataSourceOutputsRecall = [](HistogramDataSource &dataSource) {
            dataSource.viewOutput = ~0u;
            dataSource.recallOutput = 0;
            dataSource.imageOutput = 1;
        };
        //setDataSourceOutputsRecall(testDataSource);
        //setDataSourceOutputsRecall(trainingDataSource);
        auto setDataSourceOutputsView = [](HistogramDataSource &dataSource) {
            dataSource.viewOutput = 0;
            dataSource.recallOutput = ~0u;
            dataSource.imageOutput = 1;
        };
        setDataSourceOutputsView(testDataSource);
        setDataSourceOutputsView(trainingDataSource);


        convnet::InputLayer *inputLayer = net.getLayer<convnet::InputLayer>("input");
        inputLayer->setInputSizes({
            inputSize,
            convnet::TensorSize(1, 1, 1, 1)
        });

        auto computeStream = net.allocateExecutionStream();

        net.resizeLayers(*computeStream, 100);
        
        std::cout << "Output size: " 
            << net.getConnectionList().outputs[net.getConnectionList().findOutput("inferredLabel")].size << std::endl;
        
        convnet::WholeImageClassificationTrainer trainer(net, trainingDataSource);
        
        net.pushParametersToBackend(trainer.getExecutionWorkspace(), trainer.getComputeStream());

        std::vector<float> accuracies;

        unsigned burninIters = 100;
        unsigned currentBurninIter = 0;

        float initialLearningRate = -0.02f;
        float learningRate = initialLearningRate;
        
        std::vector<float> trainingError;
        std::vector<float> testError;
        std::vector<float> lastAccuracyVsRank;
        std::vector<float> lastAccuracyVsRankTrain;
        std::vector<float> trainingErrorAUC;
        std::vector<float> testErrorAUC;
        
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
            
            auto putArray = [](std::ostream &stream, const std::string &varName, const std::string &name, bool secondAxis, const std::vector<float> &array) {
                stream << "var " << varName << " = { y: [";
                for (unsigned i = 0; i < array.size(); i++) 
                    if (i > 0)
                        stream << ", " << array[i];
                    else
                        stream << ' ' << array[i];
                stream << "], name: \""<<name<<"\", ";

                if (secondAxis)
                    stream << "yaxis: 'y2', ";

                stream << "type: 'scatter'};" << std::endl;
            };

            putArray(main, "trainingError", "training error", false, trainingError);
            putArray(main, "testError", "test accuracy (top3)", true, testError);
            putArray(main, "trainingErrorAUC", "training error (AUC)", true, trainingErrorAUC);
            putArray(main, "testErrorAUC", "test error  (AUC)", true, testErrorAUC);
            putArray(main, "lastAccuracyVsRank", "accuracy vs rank (test)", false, lastAccuracyVsRank);
            putArray(main, "lastAccuracyVsRankTrain", "accuracy vs rank (train)", false, lastAccuracyVsRankTrain);


            main
                    << "var data = [trainingError, testError, trainingErrorAUC, testErrorAUC];" << std::endl
                    << "var layout = {title: 'Training', yaxis: {title: 'Training error'}, yaxis2: { title: 'Test accuracy top 3', overlaying: 'y', side: 'right'}};" << std::endl
                    << "Plotly.newPlot('chartDiv1', data, layout);" << std::endl;
            main
                    << "var data2 = [lastAccuracyVsRank, lastAccuracyVsRankTrain];" << std::endl
                    << "var layout2 = {title: 'Training', yaxis: {title: 'Accuracy'}, xaxis: {title: 'Rank'}};" << std::endl
                    << "Plotly.newPlot('chartDiv2', data2, layout2);" << std::endl;
            main
                    << "</script>" << std::endl
                << "</body>" << std::endl
                << "</html>" << std::endl;
            httpReport.putText(main, "index_eye_hist.html");
        };
        
        
        // 100 epochs: AUC: 0.70775
        for (unsigned epoch = 0; epoch < 100; epoch++) {
    
            if (epoch == 50) {
                setDataSourceOutputsRecall(testDataSource);
                setDataSourceOutputsRecall(trainingDataSource);
                learningRate = initialLearningRate;
            }

            float sumTrainingError = 0.0f;
            
            trainingDataSource.restart();
            unsigned iter = 0;        
            
            while (true) {
                std::cout << "epoch " << epoch << std::endl;
                std::cout << "        iter " << iter << std::endl;
                std::cout << "            learningRate " << learningRate << std::endl;

                if (currentBurninIter < burninIters) {
                    currentBurninIter++;
                    std::cout << "Burnin lr " << learningRate * std::exp(((float)currentBurninIter - (float)burninIters) * 0.1f) << std::endl;
                    if (!trainer.learn(1, learningRate * std::exp(((float)currentBurninIter - (float)burninIters) * 0.1f))) break;
                } else
                    if (!trainer.learn(1, learningRate)) break;
                    
                sumTrainingError += trainer.getLastBatchLabelError();


                iter++;
            }
#if 1
        if (epoch % 10 == 0) {
            std::cout << "Running validation" << std::endl;
            Engine::CPUStopWatch stopWatch;
            
            std::vector<float> accuracyVsRank;
            trainer.runValidation(testDataSource, accuracyVsRank);
            lastAccuracyVsRank = accuracyVsRank;

            std::vector<float> accuracyVsRankTrain;
            trainer.runValidation(trainingDataSource, accuracyVsRankTrain);
            lastAccuracyVsRankTrain = accuracyVsRankTrain;
            
            {
                float AUC = 0.0f;
                for (float f : accuracyVsRank)
                    AUC += f;
                AUC /= accuracyVsRank.size();
                testErrorAUC.push_back(AUC);
            }
            {
                float AUC = 0.0f;
                for (float f : accuracyVsRankTrain)
                    AUC += f;
                AUC /= accuracyVsRankTrain.size();
                trainingErrorAUC.push_back(AUC);
            }
            
            std::cout << "accuracyVsRank = [";
            for (float f : accuracyVsRank)
                std::cout << " " << f;
            std::cout << "];" << std::endl;
            

            trainingError.push_back(sumTrainingError);
            std::cout << "trainingError = [";
            for (float f : trainingError)
                std::cout << " " << f;
            std::cout << "];" << std::endl;

            testError.push_back(accuracyVsRank[2]);
            std::cout << "testError = [";
            for (float f : testError)
                std::cout << " " << f;
            std::cout << "];" << std::endl;

            updateReport(epoch+1, 0);
            
            trainer.report(testDataSource, "outputs_eye_test", httpReport);
            trainer.report(trainingDataSource, "outputs_eye_train", httpReport);

            std::cout << "Running validation took " << stopWatch.getNanoseconds() * 1e-9f << " seconds!" << std::endl;
        }
#endif
            
            
            if (epoch % 10 == 9)
                learningRate *= 0.5f;
        }
        {
            std::cout << "Running validation" << std::endl;
            Engine::CPUStopWatch stopWatch;

            trainer.runValidation(testDataSource, subjectResults[droppedSubject].accuracyVsRank, &subjectResults[droppedSubject].confusion, &subjectResults[droppedSubject].confusion_top3);
            lastAccuracyVsRank = subjectResults[droppedSubject].accuracyVsRank;

            float AUC = 0.0f;
            for (float f : subjectResults[droppedSubject].accuracyVsRank)
                AUC += f;
            AUC /= subjectResults[droppedSubject].accuracyVsRank.size();
            testErrorAUC.push_back(AUC);
            
            
            subjectResults[droppedSubject].top1 = subjectResults[droppedSubject].accuracyVsRank[0];
            subjectResults[droppedSubject].top3 = subjectResults[droppedSubject].accuracyVsRank[2];
            subjectResults[droppedSubject].AUC = AUC;

			averageAUC += AUC;
			averageDenom ++;
            
            totalConfusion += subjectResults[droppedSubject].confusion;
            totalConfusion_top3 += subjectResults[droppedSubject].confusion_top3;

            
            testError.push_back(subjectResults[droppedSubject].accuracyVsRank[2]);
            std::cout << "testError = [";
            for (float f : testError)
                std::cout << " " << f;
            std::cout << "];" << std::endl;

            
            updateResults();
            
            trainer.report(testDataSource, "outputs_eye_test", httpReport);
            trainer.report(trainingDataSource, "outputs_eye_train", httpReport);

            std::cout << "Running validation took " << stopWatch.getNanoseconds() * 1e-9f << " seconds!" << std::endl;
        }
    }
}
