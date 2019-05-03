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

#include "HistogramDataSource.h"
#include "ImageClassificationMatrixPlots.h"
#include "ViewRecallMatchingHistogramTripletLossCrossVal.h"
#include "ViewRecallMatchingHistogramTripletLossCrossValIndividual.h"
#include "ViewRecallMatchingHistogramTripletLossExportDescriptors.h"


#include <tools/TaskScheduler.h>


#include <eyeTracking/Dataset.h>

#include <boost/format.hpp>

#include <cudaUtils/CudaDriver.h>
#include <cudaUtils/CudaDevice.h>
#include <cudaUtils/CudaDeviceContext.h>
#include <cudaUtils/CudaProfilingScope.h>

#include <convnet/backend/CuDnnWrapper.h>


#include <iostream>
#include <random>

int main(int argc, char **argv)
{
    TaskScheduler::Init(8);

    CudaUtils::CudaDriver cudaDriver;
    std::unique_ptr<CudaUtils::CudaDevice> cudaDevice(cudaDriver.getMaxCCDevice());

    if (cudaDevice == nullptr) {
        std::cout << "No compute device found!" << std::endl;
        return -1;
    }

    std::cout << "Using device " << cudaDevice->getDeviceName() << std::endl;
    std::unique_ptr<CudaUtils::CudaDeviceContext> context(new CudaUtils::CudaDeviceContext(cudaDevice.get()));
    context->setBankSize4Byte();
    context->setPreferredL1Cache();
    context->makeCurrent();
    std::cout << "Cuda context set up" << std::endl;

    convnet::CuDnnWrapper cudnnWrapper;

    eyeTracking::Dataset dataset;
    dataset.loadStimuli(
        "../../../stimuli/iccv/",
        "../../../stimuli/iccv_category.txt",
        "../../../stimuli/iccv_labels.csv"
    );


    for (const auto &img : dataset.getImages()) {
        std::cout << "Img: " << img.filename << "   " << dataset.getCategories()[img.category].name << std::endl;
    }

#if 0
    dataset.loadAllTestSubjects("../../../data/iccv/");

/*
    ScanPathData data;
    data.sampleFromDataset(dataset, 4096, 1.0f, true, true);
    std::cout << "Got " << data.scanPaths.size() << " scan paths!" << std::endl;
    data.writeToFile("data_4096_1.0_all.bin");
*/
	
    HistogramData data;
    data.sampleFromDataset(dataset, 24, 24, true, true, 0, false, true);
    data.writeToFile("data_hist_24x24_normalized_histogram.bin");

    HistogramData dataTest;
/*    
    dataTest.sampleFromDataset(dataset, 24, 24, true, true, 0, true);
    dataTest.writeToFile("data_hist_24x24_view_merged_into_recall.bin");
 */  
    return 0;
#else
/*
    ScanPathData data;
    data.readFromFile("../buildRel/data_4096_1.0_all.bin");
*/
    HistogramData data;
//    data.readFromFile("../buildRel/data_hist_36x36_all.bin");
  	//data.readFromFile("../buildRel/data_hist_24x24_all.bin");
//    data.readFromFile("../buildRel/data_hist_24x24_view_merged_into_recall.bin");

    HistogramData dataTest;
    //dataTest.readFromFile("../buildRel/data_hist_24x24_all_augment0.bin");
	dataTest.readFromFile("../buildRel/data_hist_24x24_no_augment.bin");
	//dataTest.readFromFile("../buildRel/data_hist_24x24_normalized_histogram.bin");
#endif    
    
//  	plotView(dataset);
//  	plotMovement(dataset);

//    trainCategoryClassification(dataset, cudnnWrapper);
//    trainViewRecallMatching(dataset, data, cudnnWrapper);
 
    //trainViewRecallMatchingHistogram(dataset, data, dataTest, cudnnWrapper);
//trainViewRecallMatchingHistogram(dataset, dataTest, dataTest, cudnnWrapper);
//trainViewRecallMatchingHistogramTripletLossCrossVal(dataset, dataTest, dataTest, cudnnWrapper);
trainViewRecallMatchingHistogramTripletLossCrossValIndividual(dataset, dataTest, dataTest, cudnnWrapper);
//trainViewRecallMatchingHistogramTripletLossTraining(dataset, dataTest, dataTest, cudnnWrapper);
//trainViewRecallMatchingHistogramTripletLossExportDescriptors(dataset, dataTest, dataTest, cudnnWrapper);
    
  //  trainImageClassification(dataset, data, cudnnWrapper);
	//imageClassificationmatrixPlots(dataset, data, dataTest, cudnnWrapper);
	//imageClassificationmatrixPlots(dataset, dataTest, dataTest, cudnnWrapper);


    return 0;

}



