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

#define BOOST_TEST_MODULE "Unit tests for the genModels lib"
#include <boost/test/unit_test.hpp>

#include "LayerTest.h"

#include <tools/TaskScheduler.h>

#include <cudaUtils/CudaDriver.h>
#include <cudaUtils/CudaDevice.h>
#include <cudaUtils/CudaDeviceContext.h>

#include <convnet/backend/ConvNetImpl_cudnn.h>
#include <convnet/backend/CuDnnWrapper.h>

#include <convnet/layers/InputLayer.h>
#include <convnet/layers/PoolingLayer.h>
//#include <convnet/layers/UnpoolingLayer.h>
//#include <convnet/layers/ReluSplitLayer.h>
//#include <convnet/layers/ConcatenationLayer.h>
#include <convnet/layers/ReluLayer.h>
//#include <convnet/layers/PReluLayer.h>
//#include <convnet/layers/ElemWiseOpLayer.h>
//#include <convnet/layers/SaturationLayer.h>
//#include <convnet/layers/CrossEntropyLossLayer.h>
//#include <convnet/layers/ReconstructionLossLayer.h>
#include <convnet/layers/DropoutLayer.h>
//#include <convnet/layers/LocalResponseNormalizationLayer.h>
//#include <convnet/layers/BatchVariationLayer.h>
//#include <convnet/layers/ConcatInsetLayer.h>
//#include <convnet/layers/CropLayer.h>
#include <convnet/layers/ConvolutionLayer.h>
//#include <convnet/layers/SpatialTransformerLayer.h>
//#include <convnet/layers/BroadcastLayer.h>
//#include <convnet/layers/ReductionLayer.h>

#include <random>

#include <iostream>

struct TaskSchedulerSetup {
    TaskSchedulerSetup() {
        TaskScheduler::Init(8);
    }
    ~TaskSchedulerSetup()  {
        TaskScheduler::Shutdown();
    }
};

struct CudaSetup {
    static CudaSetup *cudaSetup;

    CudaUtils::CudaDriver cudaDriver;
    std::unique_ptr<CudaUtils::CudaDevice> cudaDevice;
    std::unique_ptr<CudaUtils::CudaDeviceContext> context;
    std::unique_ptr<convnet::CuDnnWrapper> cudnnWrapper;
    CudaSetup() {
        cudaSetup = this;
        cudaDevice.reset(cudaDriver.getMaxCCDevice());

        if (cudaDevice == nullptr) {
            throw std::runtime_error("No compute device found!");
        }

        std::cout << "Using device " << cudaDevice->getDeviceName() << std::endl;
        context.reset(new CudaUtils::CudaDeviceContext(cudaDevice.get()));
        context->setBankSize4Byte();
        context->setPreferredL1Cache();
        context->makeCurrent();
        std::cout << "Cuda context set up" << std::endl;

        cudnnWrapper.reset(new convnet::CuDnnWrapper());
    }
    ~CudaSetup()  {
        cudaSetup = nullptr;
    }
};

CudaSetup *CudaSetup::cudaSetup = nullptr;


BOOST_GLOBAL_FIXTURE( TaskSchedulerSetup );
BOOST_GLOBAL_FIXTURE( CudaSetup );

template<class LayerSetupFunctor>
void testDataDerivativeSimple(const convnet::TensorSize &inputSize, unsigned batchSize, LayerSetupFunctor layerSetup)
{
    std::mt19937 rng;

    convnet::ConvNetImpl_cudnn net(*CudaSetup::cudaSetup->cudnnWrapper);
    net.addLayer<convnet::InputLayer>("input")->setOutputConnectionNames({""});
    layerSetup(net);

    net.connectLayers();

    convnet::InputLayer *inputLayer = net.getLayer<convnet::InputLayer>("input");
    inputLayer->setInputSizes({inputSize});

    DataDerivativeTest test(net, batchSize);
    test.testAll();
}

#if 0
BOOST_AUTO_TEST_CASE(testCrossEntropyLossMulti)
{
    std::mt19937 rng;
    std::normal_distribution<float> normalDist;
    std::uniform_int_distribution<unsigned> labelDist(0, 5);

    convnet::ConvNetImpl_cudnn net(*CudaSetup::cudaSetup->cudnnWrapper);
    net.addLayer<convnet::InputLayer>("input")
        ->setOutputConnectionNames({"data", "label"});

    net.addLayer<convnet::CrossEntropyLossLayer>()
        ->setInputConnectionNames({"data", "label"});

    net.connectLayers();

    convnet::InputLayer *inputLayer = net.getLayer<convnet::InputLayer>("input");
    inputLayer->setInputSizes({convnet::TensorSize(2, 3, 4, 5), convnet::TensorSize(2, 3, 4, 1)});

    LossLayerTest test(net, 8, [&](unsigned inputIndex)->float{
        if (inputIndex == 0) return normalDist(rng);
        unsigned l = labelDist(rng);
        return (l == 5)?0xFF:l;
    });
    test.testAll();    
}


BOOST_AUTO_TEST_CASE(testCrossEntropyLossScalar)
{
    std::mt19937 rng;
    std::normal_distribution<float> normalDist;
    std::uniform_int_distribution<unsigned> labelDist(0, 5);

    convnet::ConvNetImpl_cudnn net(*CudaSetup::cudaSetup->cudnnWrapper);
    net.addLayer<convnet::InputLayer>("input")
        ->setOutputConnectionNames({"data", "label"});

    net.addLayer<convnet::CrossEntropyLossLayer>()
        ->setInputConnectionNames({"data", "label"});

    net.connectLayers();

    convnet::InputLayer *inputLayer = net.getLayer<convnet::InputLayer>("input");
    inputLayer->setInputSizes({convnet::TensorSize(1, 1, 1, 5), convnet::TensorSize(1, 1, 1, 1)});

    LossLayerTest test(net, 32, [&](unsigned inputIndex)->float{
        if (inputIndex == 0) return normalDist(rng);
        unsigned l = labelDist(rng);
        return (l == 5)?0xFF:l;
    });
    test.testAll();
}


BOOST_AUTO_TEST_CASE(testReconstructionLoss)
{
    std::mt19937 rng;
    std::normal_distribution<float> normalDist;

    convnet::ConvNetImpl_cudnn net(*CudaSetup::cudaSetup->cudnnWrapper);
    net.addLayer<convnet::InputLayer>("input")
        ->setOutputConnectionNames({"data", "ref"});

    net.addLayer<convnet::ReconstructionLossLayer>()
        ->setLossType(convnet::ReconstructionLossLayer::LT_L2)
        ->setInputConnectionNames({"data", "ref"});

    net.connectLayers();

    convnet::InputLayer *inputLayer = net.getLayer<convnet::InputLayer>("input");
    inputLayer->setInputSizes({convnet::TensorSize(2, 3, 4, 5), convnet::TensorSize(2, 3, 4, 5)});

    LossLayerTest test(net, 32, [&](unsigned /*inputIndex*/)->float{
        return normalDist(rng);
    });
    test.testAll();    
}


BOOST_AUTO_TEST_CASE(testReluSplitLayerDataDerivative)
{
    testDataDerivativeSimple(convnet::TensorSize(2, 3, 4, 5), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::ReluSplitLayer>();
    });
}
#endif
BOOST_AUTO_TEST_CASE(testReluLayerDataDerivative)
{
    testDataDerivativeSimple(convnet::TensorSize(2, 3, 4, 5), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::ReluLayer>();
    });
}
#if 0
BOOST_AUTO_TEST_CASE(testSaturationLayerDataDerivative)
{
    testDataDerivativeSimple(convnet::TensorSize(2, 3, 4, 5), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::SaturationLayer>();
    });
}


BOOST_AUTO_TEST_CASE(CropLayerDataDerivative)
{
    testDataDerivativeSimple(convnet::TensorSize(5, 5, 2, 8), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::CropLayer>()->cropCenter(Eigen::Vector3i(2, 2, 1));
    });
}



BOOST_AUTO_TEST_CASE(BroadcastLayerDataDerivative)
{
    testDataDerivativeSimple(convnet::TensorSize(1, 5, 2, 8), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::BroadcastLayer>()->setBroadcast(5, 0, 0, 0);
    });
    testDataDerivativeSimple(convnet::TensorSize(5, 1, 2, 8), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::BroadcastLayer>()->setBroadcast(0, 5, 0, 0);
    });
    testDataDerivativeSimple(convnet::TensorSize(5, 5, 1, 8), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::BroadcastLayer>()->setBroadcast(0, 0, 2, 0);
    });
    testDataDerivativeSimple(convnet::TensorSize(5, 5, 2, 1), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::BroadcastLayer>()->setBroadcast(0, 0, 0, 8);
    });
    testDataDerivativeSimple(convnet::TensorSize(1, 1, 2, 8), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::BroadcastLayer>()->setBroadcast(5, 5, 0, 0);
    });
    testDataDerivativeSimple(convnet::TensorSize(1, 1, 2, 1), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::BroadcastLayer>()->setBroadcast(5, 5, 0, 8);
    });
}

BOOST_AUTO_TEST_CASE(ReductionLayerDataDerivative)
{
    testDataDerivativeSimple(convnet::TensorSize(5, 5, 2, 8), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::ReductionLayer>()->setReduction(true, false, false, false);
    });
    testDataDerivativeSimple(convnet::TensorSize(5, 5, 2, 8), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::ReductionLayer>()->setReduction(false, true, false, false);
    });
    testDataDerivativeSimple(convnet::TensorSize(5, 5, 2, 8), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::ReductionLayer>()->setReduction(false, false, true, false);
    });
    testDataDerivativeSimple(convnet::TensorSize(5, 5, 2, 8), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::ReductionLayer>()->setReduction(false, false, false, true);
    });

    testDataDerivativeSimple(convnet::TensorSize(5, 5, 2, 8), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::ReductionLayer>()->setReduction(true, true, false, false);
    });
    testDataDerivativeSimple(convnet::TensorSize(5, 5, 2, 8), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::ReductionLayer>()->setReduction(true, true, false, true);
    });
}




BOOST_AUTO_TEST_CASE(testPReluLayerDataDerivative)
{
    std::mt19937 rng;
    testDataDerivativeSimple(convnet::TensorSize(2, 3, 4, 5), 2, [&rng](convnet::ConvNet &net){
        net.addLayer<convnet::PReluLayer>()->seedParametersGaussian(5, rng);
    });
}
#endif
BOOST_AUTO_TEST_CASE(testMaxPoolLayerDataDerivative)
{
    testDataDerivativeSimple(convnet::TensorSize(2, 3, 4, 5), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::PoolingLayer>()->setMode(convnet::PoolingLayer::MODE_MAX);
    });
}

BOOST_AUTO_TEST_CASE(testAvgPoolLayerDataDerivative)
{
    testDataDerivativeSimple(convnet::TensorSize(2, 3, 4, 5), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::PoolingLayer>()->setMode(convnet::PoolingLayer::MODE_AVG);
    });
}
#if 0
BOOST_AUTO_TEST_CASE(testAvgUnpoolLayerDataDerivative)
{
    testDataDerivativeSimple(convnet::TensorSize(4, 4, 9, 5), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::UnpoolingLayer>()->setWindowSize(2, 2, 3);
    });
}

BOOST_AUTO_TEST_CASE(testLRNLayerDataDerivative)
{
    testDataDerivativeSimple(convnet::TensorSize(4, 4, 9, 5), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::LocalResponseNormalizationLayer>();
    });
}
#endif
/*
 * Does not work, because the test assumes the instances are independent.
BOOST_AUTO_TEST_CASE(testBatchVariationLayerDataDerivative)
{
    testDataDerivativeSimple(convnet::TensorSize(1, 1, 1, 1), 2, [](convnet::ConvNet &net){
        net.addLayer<convnet::BatchVariationLayer>();
    });
}
*/

#if 0
BOOST_AUTO_TEST_CASE(testConcatLayerSameSizeDataDerivative)
{
    std::mt19937 rng;

    convnet::ConvNetImpl_cudnn net(*CudaSetup::cudaSetup->cudnnWrapper);
    net.addLayer<convnet::InputLayer>("input")->setOutputConnectionNames({"a", "b"});
    net.addLayer<convnet::ConcatenationLayer>()->setInputConnectionNames({"a", "b"});

    net.connectLayers();

    convnet::InputLayer *inputLayer = net.getLayer<convnet::InputLayer>("input");
    inputLayer->setInputSizes({convnet::TensorSize(1, 1, 1, 5), convnet::TensorSize(1, 1, 1, 3)});

    DataDerivativeTest test(net, 2);
    test.testAll();
}

BOOST_AUTO_TEST_CASE(testConcatLayerDifferentSizeDataDerivative_1)
{
    std::mt19937 rng;

    convnet::ConvNetImpl_cudnn net(*CudaSetup::cudaSetup->cudnnWrapper);
    net.addLayer<convnet::InputLayer>("input")->setOutputConnectionNames({"a", "b"});
    net.addLayer<convnet::ConcatenationLayer>()->setInputConnectionNames({"a", "b"});

    net.connectLayers();

    convnet::InputLayer *inputLayer = net.getLayer<convnet::InputLayer>("input");
    inputLayer->setInputSizes({convnet::TensorSize(5, 5, 5, 5), convnet::TensorSize(2, 3, 4, 3)});

    DataDerivativeTest test(net, 2);
    test.testAll();
}

BOOST_AUTO_TEST_CASE(testConcatLayerDifferentSizeDataDerivative_2)
{
    std::mt19937 rng;

    convnet::ConvNetImpl_cudnn net(*CudaSetup::cudaSetup->cudnnWrapper);
    net.addLayer<convnet::InputLayer>("input")->setOutputConnectionNames({"a", "b"});
    net.addLayer<convnet::ConcatenationLayer>()->setInputConnectionNames({"a", "b"});

    net.connectLayers();

    convnet::InputLayer *inputLayer = net.getLayer<convnet::InputLayer>("input");
    inputLayer->setInputSizes({convnet::TensorSize(2, 3, 4, 3), convnet::TensorSize(5, 5, 5, 5)});

    DataDerivativeTest test(net, 2);
    test.testAll();
}

BOOST_AUTO_TEST_CASE(testConcatInsetLayerDataDerivative)
{
    std::mt19937 rng;

    convnet::ConvNetImpl_cudnn net(*CudaSetup::cudaSetup->cudnnWrapper);
    net.addLayer<convnet::InputLayer>("input")->setOutputConnectionNames({"a", "b"});
    net.addLayer<convnet::ConcatInsetLayer>()->setInputConnectionNames({"a", "b"});

    net.connectLayers();

    convnet::InputLayer *inputLayer = net.getLayer<convnet::InputLayer>("input");
    inputLayer->setInputSizes({convnet::TensorSize(5, 5, 5, 5), convnet::TensorSize(2, 2, 2, 3)});

    DataDerivativeTest test(net, 2);
    test.testAll();
}


BOOST_AUTO_TEST_CASE(testElemWiseOpLayerDataDerivative)
{
    std::mt19937 rng;

    convnet::ConvNetImpl_cudnn net(*CudaSetup::cudaSetup->cudnnWrapper);
    net.addLayer<convnet::InputLayer>("input")->setOutputConnectionNames({"a", "b"});
    net.addLayer<convnet::ConcatenationLayer>()->setInputConnectionNames({"a", "b"});

    net.connectLayers();

    convnet::InputLayer *inputLayer = net.getLayer<convnet::InputLayer>("input");
    inputLayer->setInputSizes({convnet::TensorSize(2, 2, 2, 5), convnet::TensorSize(2, 1, 2, 5)});

    DataDerivativeTest test(net, 2);
    test.testAll();
}




BOOST_AUTO_TEST_CASE(testSpatialDropoutLayer)
{
    std::mt19937 rng;

    float drop = 0.7f;
    
    convnet::ConvNetImpl_cudnn net(*CudaSetup::cudaSetup->cudnnWrapper);
    net.addLayer<convnet::InputLayer>("input")->setOutputConnectionNames({"a"});
    net.addLayer<convnet::DropoutLayer>()
            ->setDropout(drop)
            ->setSpatialDropout(true);
            

    net.connectLayers();

    convnet::InputLayer *inputLayer = net.getLayer<convnet::InputLayer>("input");
    inputLayer->setInputSizes({convnet::TensorSize(8, 8, 2, 128)});

        
    std::unique_ptr<convnet::ExecutionStream> executionStream = net.allocateExecutionStream();
    net.resizeLayers(*executionStream, 64);
    std::unique_ptr<convnet::ExecutionWorkspace> executionWorkspace = net.allocateExecutionWorkspace();

    
    net.pushParametersToBackend(*executionWorkspace, *executionStream);

    convnet::NetworkState states = net.allocateState(*executionStream);

    std::uniform_real_distribution<float> randomNumber(1.0f, 2.0f);
    
    
    states.clear(*executionStream);

    
    {
        convnet::TensorData &output = dynamic_cast<convnet::TensorData&>(*states.outputs.front());

        convnet::MappedTensor mappedInput = output.getValues().lock(true);
        const convnet::TensorSize &inputSize = mappedInput.getSize();
        for (unsigned n = 0; n < mappedInput.getNumInstances(); n++)
            for (unsigned c = 0; c < inputSize.channels; c++)
                for (unsigned z = 0; z < inputSize.depth; z++)
                    for (unsigned y = 0; y < inputSize.height; y++)
                        for (unsigned x = 0; x < inputSize.width; x++) {
                            mappedInput.get<float>(x, y, z, c, n) = randomNumber(rng);
                        }

        output.getValues().unlock(mappedInput);
    }
    
    net.feedForward(states, *executionWorkspace, *executionStream, true);
    executionStream->flush();

    {
        convnet::TensorData &input = dynamic_cast<convnet::TensorData&>(*states.outputs.front());
        convnet::TensorData &output = dynamic_cast<convnet::TensorData&>(*states.outputs.back());

        convnet::MappedTensor mappedInput = input.getValues().lock();
        convnet::MappedTensor mappedOutput = output.getValues().lock();
        const convnet::TensorSize &inputSize = mappedInput.getSize();
        unsigned numBlank = 0;
        for (unsigned n = 0; n < mappedInput.getNumInstances(); n++)
            for (unsigned c = 0; c < inputSize.channels; c++) {
                bool isBlank;
                for (unsigned z = 0; z < inputSize.depth; z++)
                    for (unsigned y = 0; y < inputSize.height; y++)
                        for (unsigned x = 0; x < inputSize.width; x++) {
                            float i = mappedInput.get<float>(x, y, z, c, n);
                            float o = mappedOutput.get<float>(x, y, z, c, n);
                            
                            if (x == 0 && y == 0 && z == 0)
                                isBlank = o == 0.0f;
                            else 
                                BOOST_CHECK_MESSAGE((o == 0.0f) == isBlank, "Incoherent dropout");

                            if (o != 0.0f) 
                                BOOST_CHECK_MESSAGE((i != 0.0f), "Broken forwarding");
                        }
                if (isBlank) numBlank++;
            }
            
        std::cout << "Num blank: " << numBlank << " / " << mappedInput.getNumInstances()*inputSize.channels << " should be " << drop * mappedInput.getNumInstances()*inputSize.channels << std::endl;

        input.getValues().unlock(mappedInput);
        output.getValues().unlock(mappedOutput);
    }

    
    {
        convnet::TensorData &output = dynamic_cast<convnet::TensorData&>(*states.outputs.back());

        convnet::MappedTensor mappedInput = output.getGradients().lock(true);
        const convnet::TensorSize &inputSize = mappedInput.getSize();
        for (unsigned n = 0; n < mappedInput.getNumInstances(); n++)
            for (unsigned c = 0; c < inputSize.channels; c++)
                for (unsigned z = 0; z < inputSize.depth; z++)
                    for (unsigned y = 0; y < inputSize.height; y++)
                        for (unsigned x = 0; x < inputSize.width; x++) {
                            mappedInput.get<float>(x, y, z, c, n) = randomNumber(rng);
                        }

        output.getGradients().unlock(mappedInput);
    }
    
    net.feedBackward(states, *executionWorkspace, *executionStream, false);
    executionStream->flush();

    {
        convnet::TensorData &input = dynamic_cast<convnet::TensorData&>(*states.outputs.front());
        convnet::TensorData &output = dynamic_cast<convnet::TensorData&>(*states.outputs.back());

        convnet::MappedTensor mappedInput = input.getGradients().lock();
        convnet::MappedTensor mappedOutputVal = output.getValues().lock();
        //convnet::MappedTensor mappedOutput = output.getGradients().lock();
        const convnet::TensorSize &inputSize = mappedInput.getSize();
        for (unsigned n = 0; n < mappedInput.getNumInstances(); n++)
            for (unsigned c = 0; c < inputSize.channels; c++) {
                bool isBlank;
                for (unsigned z = 0; z < inputSize.depth; z++)
                    for (unsigned y = 0; y < inputSize.height; y++)
                        for (unsigned x = 0; x < inputSize.width; x++) {
                            float i = mappedInput.get<float>(x, y, z, c, n);
                            //float o = mappedOutput.get<float>(x, y, z, c, n);
                            float v = mappedOutputVal.get<float>(x, y, z, c, n);
                            
                            if (x == 0 && y == 0 && z == 0)
                                isBlank = v == 0.0f;
                            else 
                                BOOST_CHECK_MESSAGE((v == 0.0f) == isBlank, "Incoherent dropout in backward pass");

                            BOOST_CHECK_MESSAGE(!((i == 0.0f) && (v != 0.0f)), "No gradient in not-dropped connection");
                            BOOST_CHECK_MESSAGE(!((i != 0.0f) && (v == 0.0f)), "Gradient in dropped connection");
                        }
            }

        input.getGradients().unlock(mappedInput);
        output.getValues().unlock(mappedOutputVal);
        //output.getGradients().unlock(mappedOutput);
    }
    
}



BOOST_AUTO_TEST_CASE(testTransformLayer)
{
    std::mt19937 rng;

    convnet::ConvNetImpl_cudnn net(*CudaSetup::cudaSetup->cudnnWrapper);
    net.addLayer<convnet::InputLayer>("input")->setOutputConnectionNames({"a", "b"});
    convnet::ConvolutionLayer *convLayer = net.addLayer<convnet::ConvolutionLayer>();
    convLayer->setInputConnectionNames({"b"})->setOutputConnectionNames({"c"});
    convLayer->seedParametersGaussian(6, convnet::TensorSize(1, 1, 1, 6), rng, 0.0001f, 0.00001f);
    
    convLayer->getParameters()->m_kernel[0].bias = 1.0f;
    convLayer->getParameters()->m_kernel[1].bias = 0.0f;
    convLayer->getParameters()->m_kernel[2].bias = 0.0f;
    convLayer->getParameters()->m_kernel[3].bias = 0.0f;
    convLayer->getParameters()->m_kernel[4].bias = 1.0f;
    convLayer->getParameters()->m_kernel[5].bias = 0.0f;
    for (unsigned i = 0; i < 6; i++)
        convLayer->getParameters()->m_kernel[i].filter.setZero();
          
    net.addLayer<convnet::SpatialTransformerLayer>()->setInputConnectionNames({"a", "c"});
    net.addLayer<convnet::CropLayer>()->cropCenter(Eigen::Vector3i(16, 16, 1));

    net.connectLayers();

    convnet::InputLayer *inputLayer = net.getLayer<convnet::InputLayer>("input");
    inputLayer->setInputSizes({convnet::TensorSize(32, 32, 1, 5), convnet::TensorSize(1, 1, 1, 6)});

    DataDerivativeTest test(net, 2);
    test.testAll();
}

BOOST_AUTO_TEST_CASE(testConvolutionOrthonormalRegularization)
{
    std::mt19937 rng;

    convnet::ConvNetImpl_cudnn net(*CudaSetup::cudaSetup->cudnnWrapper);
    convnet::InputLayer *inputLayer = net.addLayer<convnet::InputLayer>("input");
    inputLayer->setOutputConnectionNames({"a"});
    
    inputLayer->setInputSizes({convnet::TensorSize(1, 1, 1, 4)});

    convnet::ConvolutionLayer *convLayer = net.addLayer<convnet::ConvolutionLayer>();
    convLayer->seedParametersGaussian(2, convnet::TensorSize(1, 1, 1, 4), rng, 0.1f, 0.1f);
    convLayer->getParameters()->m_orthoNormalDecay = 0.01f;
    convLayer->getParameters()->m_weightDecay = 0.0f;
    //convLayer->setParameterLambdas(0.2f, 0.2f);

    net.connectLayers();
    
    
    auto executionStream = net.allocateExecutionStream();
    net.resizeLayers(*executionStream, 4);
    auto executionWorkspace = net.allocateExecutionWorkspace();

    /*
    auto state = net.allocateState(*executionStream);
    state.clear(*executionStream);
    */
    
    net.pushParametersToBackend(*executionWorkspace, *executionStream);
    
    auto printParameters = [&]{
        executionStream->flush();
        net.pullParametersFromBackend(*executionWorkspace, *executionStream);
        
        for (unsigned i = 0; i < convLayer->getParameters()->m_kernel.size(); i++) {
            std::cout << "Kernel " << i << std::endl;
            std::cout << "  " << convLayer->getParameters()->m_kernel[i].bias << std::endl;
            for (unsigned j = 0; j < convLayer->getParameters()->m_filterSize.numElements(); j++)
                std::cout << "  " << convLayer->getParameters()->m_kernel[i].filter[j];
            std::cout << std::endl;
        }
        
    };
    
    printParameters();
    for (unsigned i = 0; i < 500; i++) {
        std::cout << i << std::endl;
        convLayer->getParameters()->performParameterStep(-0.01f, *executionWorkspace, *executionStream);
        printParameters();
    }
    
}
#endif
