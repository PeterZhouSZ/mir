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

#include "LayerTest.h"

#include <boost/test/unit_test.hpp>

#include <convnet/layers/InputLayer.h>

#include <random>

boost::test_tools::predicate_result compareGradients(float finite, const char *finiteName, float analytical, const char *analyticalName, float percentageError, float absoluteCutoff)
{
    float absFinite = std::abs(finite);
    float absAnalytical = std::abs(analytical);

    if ((absFinite < absoluteCutoff) && (absFinite < absoluteCutoff))
        return true;

    float max = std::max(absFinite, absAnalytical);
    float diff = std::abs(absFinite - absAnalytical);

    float error = diff * 100.0f / max;
    if (error < percentageError)
        return true;

    boost::test_tools::predicate_result result(false);
    result.message() << "Difference exceeds limits: " << finiteName << " " << finite << " vs "
                    << analyticalName << " " << analytical << " difference is " << error << "%. (max is " << percentageError << "%)";
    return result;
}

#define COMPARE_GRADIENTS(finite, analytical, percentageError, absoluteCutoff) \
    BOOST_CHECK(compareGradients(finite, #finite, analytical, #analytical, percentageError, absoluteCutoff))




DataDerivativeTest::DataDerivativeTest(convnet::ConvNet &net, unsigned batchSize) : m_net(net)
{
    m_executionStream = net.allocateExecutionStream();
    net.resizeLayers(*m_executionStream, batchSize);
    m_executionWorkspace = net.allocateExecutionWorkspace();

    
    net.pushParametersToBackend(*m_executionWorkspace, *m_executionStream);


    m_states[0] = net.allocateState(*m_executionStream);
    m_states[1] = net.allocateState(*m_executionStream);
    m_states[2] = net.allocateState(*m_executionStream);

    std::mt19937 rng(1234);
    std::normal_distribution<float> normalDist;

    m_states[0].clear(*m_executionStream);

    convnet::InputLayer *inputLayer = m_net.getLayer<convnet::InputLayer>("input");
    {
        for (unsigned l = 0; l < inputLayer->getNumOutputs(); l++) {
            convnet::NetworkData &output = inputLayer->getOutput(m_states[0], l);

            convnet::MappedTensor mappedInput = dynamic_cast<convnet::TensorData&>(output).getValues().lock(true);
            const convnet::TensorSize &inputSize = mappedInput.getSize();
            for (unsigned n = 0; n < mappedInput.getNumInstances(); n++)
                for (unsigned c = 0; c < inputSize.channels; c++)
                    for (unsigned z = 0; z < inputSize.depth; z++)
                        for (unsigned y = 0; y < inputSize.height; y++)
                            for (unsigned x = 0; x < inputSize.width; x++) {
                                mappedInput.get<float>(x, y, z, c, n) = normalDist(rng);
                            }

            dynamic_cast<convnet::TensorData&>(output).getValues().unlock(mappedInput);
        }
    }

    net.feedForward(m_states[0], *m_executionWorkspace, *m_executionStream, true);
    m_executionStream->flush();

    {
        convnet::MappedTensor mappedOutputValues = dynamic_cast<convnet::TensorData&>(*m_states[0].outputs.back()).getValues().lock();
        convnet::MappedTensor mappedOutputGradients = dynamic_cast<convnet::TensorData&>(*m_states[0].outputs.back()).getGradients().lock(true);

        const convnet::TensorSize &outputSize = mappedOutputValues.getSize();
        for (unsigned n = 0; n < mappedOutputValues.getNumInstances(); n++)
            for (unsigned c = 0; c < outputSize.channels; c++)
                for (unsigned z = 0; z < outputSize.depth; z++)
                    for (unsigned y = 0; y < outputSize.height; y++)
                        for (unsigned x = 0; x < outputSize.width; x++) {
                            mappedOutputGradients.get<float>(x, y, z, c, n) = normalDist(rng);
                        }

        dynamic_cast<convnet::TensorData&>(*m_states[0].outputs.back()).getValues().unlock(mappedOutputValues, false);
        dynamic_cast<convnet::TensorData&>(*m_states[0].outputs.back()).getGradients().unlock(mappedOutputGradients);
    }

    net.feedBackward(m_states[0], *m_executionWorkspace, *m_executionStream, false);
    m_executionStream->flush();
}

void DataDerivativeTest::testInput(unsigned inputIndex, unsigned x, unsigned y, unsigned z, unsigned c, unsigned n)
{
    const float epsilon = 0.01f;

    m_states[1].clear(*m_executionStream);
    m_states[2].clear(*m_executionStream);
    m_executionStream->flush();
    
    convnet::InputLayer *inputLayer = m_net.getLayer<convnet::InputLayer>("input");
    for (unsigned l = 0; l < inputLayer->getNumOutputs(); l++) {
        convnet::NetworkData &output0 = inputLayer->getOutput(m_states[0], l);
        convnet::NetworkData &output1 = inputLayer->getOutput(m_states[1], l);
        convnet::NetworkData &output2 = inputLayer->getOutput(m_states[2], l);

        convnet::MappedTensor mappedInput0 = dynamic_cast<convnet::TensorData&>(output0).getValues().lock();
        convnet::MappedTensor mappedInput1 = dynamic_cast<convnet::TensorData&>(output1).getValues().lock(true);
        convnet::MappedTensor mappedInput2 = dynamic_cast<convnet::TensorData&>(output2).getValues().lock(true);
        const convnet::TensorSize &inputSize = mappedInput0.getSize();
        for (unsigned i = 0; i < mappedInput0.getNumInstances(); i++) {
            memcpy(mappedInput1.data<float>(i), mappedInput0.data<float>(i), inputSize.numElements() * sizeof(float));
            memcpy(mappedInput2.data<float>(i), mappedInput0.data<float>(i), inputSize.numElements() * sizeof(float));
        }

        if (l == inputIndex) {
            mappedInput1.get<float>(x, y, z, c, n) += epsilon;
            mappedInput2.get<float>(x, y, z, c, n) -= epsilon;
        }

        dynamic_cast<convnet::TensorData&>(output0).getValues().unlock(mappedInput0, false);
        dynamic_cast<convnet::TensorData&>(output1).getValues().unlock(mappedInput1);
        dynamic_cast<convnet::TensorData&>(output2).getValues().unlock(mappedInput2);
    }


    m_net.feedForward(m_states[1], *m_executionWorkspace, *m_executionStream, false);
    m_net.feedForward(m_states[2], *m_executionWorkspace, *m_executionStream, false);
    m_executionStream->flush();

    float finiteGradient, analyticalGradient;

    {

        convnet::MappedTensor mappedOutputValues1 = dynamic_cast<convnet::TensorData&>(*m_states[1].outputs.back()).getValues().lock();
        convnet::MappedTensor mappedOutputValues2 = dynamic_cast<convnet::TensorData&>(*m_states[2].outputs.back()).getValues().lock();
        convnet::MappedTensor mappedOutputGradients0 = dynamic_cast<convnet::TensorData&>(*m_states[0].outputs.back()).getGradients().lock();

        const convnet::TensorSize &outputSize = mappedOutputGradients0.getSize();

        finiteGradient = 0.0f;
        for (unsigned c_ = 0; c_ < outputSize.channels; c_++)
            for (unsigned z_ = 0; z_ < outputSize.depth; z_++)
                for (unsigned y_ = 0; y_ < outputSize.height; y_++)
                    for (unsigned x_ = 0; x_ < outputSize.width; x_++) {
                        float dydx = (mappedOutputValues1.get<float>(x_, y_, z_, c_, n) - mappedOutputValues2.get<float>(x_, y_, z_, c_, n)) / (2.0f * epsilon);

                        finiteGradient += dydx * mappedOutputGradients0.get<float>(x_, y_, z_, c_, n);
                    }

        dynamic_cast<convnet::TensorData&>(*m_states[1].outputs.back()).getValues().unlock(mappedOutputValues1, false);
        dynamic_cast<convnet::TensorData&>(*m_states[2].outputs.back()).getValues().unlock(mappedOutputValues2, false);
        dynamic_cast<convnet::TensorData&>(*m_states[0].outputs.back()).getGradients().unlock(mappedOutputGradients0, false);
    }
    {
        convnet::NetworkData &output0 = inputLayer->getOutput(m_states[0], inputIndex);
        convnet::MappedTensor mappedInputGradients0 = dynamic_cast<convnet::TensorData&>(output0).getGradients().lock();

        analyticalGradient = mappedInputGradients0.get<float>(x, y, z, c, n);

        dynamic_cast<convnet::TensorData&>(output0).getGradients().unlock(mappedInputGradients0, false);
    }

//    std::cout << "Input " << inputIndex << " " << x << " " << y << " " << z << " " << c << " " << n << std::endl;
//    std::cout << analyticalGradient << "    " << finiteGradient << std::endl;
    COMPARE_GRADIENTS(finiteGradient, analyticalGradient, 1.0f, 1e-5f);
}

void DataDerivativeTest::testAll()
{
    convnet::InputLayer *inputLayer = m_net.getLayer<convnet::InputLayer>("input");
    for (unsigned l = 0; l < inputLayer->getNumOutputs(); l++) {
        convnet::NetworkData &output0 = inputLayer->getOutput(m_states[0], l);
        const convnet::TensorSize &inputSize = dynamic_cast<convnet::TensorData&>(output0).getValues().getSize();

        for (unsigned n = 0; n < dynamic_cast<convnet::TensorData&>(output0).getValues().getNumInstances(); n++)
            for (unsigned c = 0; c < inputSize.channels; c++)
                for (unsigned z = 0; z < inputSize.depth; z++)
                    for (unsigned y = 0; y < inputSize.height; y++)
                        for (unsigned x = 0; x < inputSize.width; x++)
                            testInput(l, x, y, z, c, n);
    }
}






LossLayerTest::LossLayerTest(convnet::ConvNet &net, unsigned batchSize, std::function<float(unsigned)> produceValue) : m_net(net)
{
    m_executionStream = net.allocateExecutionStream();
    m_executionWorkspace = net.allocateExecutionWorkspace();

    net.resizeLayers(*m_executionStream, batchSize);

    m_states[0] = net.allocateState(*m_executionStream);
    m_states[1] = net.allocateState(*m_executionStream);
    m_states[2] = net.allocateState(*m_executionStream);


    m_states[0].clear(*m_executionStream);

    convnet::InputLayer *inputLayer = m_net.getLayer<convnet::InputLayer>("input");
    {
        for (unsigned l = 0; l < inputLayer->getNumOutputs(); l++) {
            convnet::NetworkData &output = inputLayer->getOutput(m_states[0], l);

            convnet::MappedTensor mappedInput = dynamic_cast<convnet::TensorData&>(output).getValues().lock(true);
            const convnet::TensorSize &inputSize = mappedInput.getSize();
            for (unsigned n = 0; n < mappedInput.getNumInstances(); n++)
                for (unsigned c = 0; c < inputSize.channels; c++)
                    for (unsigned z = 0; z < inputSize.depth; z++)
                        for (unsigned y = 0; y < inputSize.height; y++)
                            for (unsigned x = 0; x < inputSize.width; x++) {
                                mappedInput.get<float>(x, y, z, c, n) = produceValue(l);
                            }

            dynamic_cast<convnet::TensorData&>(output).getValues().unlock(mappedInput);
        }
    }
    net.feedForward(m_states[0], *m_executionWorkspace, *m_executionStream, true);
    m_executionStream->flush();

    net.feedBackward(m_states[0], *m_executionWorkspace, *m_executionStream, false);
    m_executionStream->flush();
}

void LossLayerTest::testInput(unsigned inputIndex, unsigned x, unsigned y, unsigned z, unsigned c, unsigned n)
{
    const float epsilon = 0.01f;

    m_states[1].clear(*m_executionStream);
    m_states[2].clear(*m_executionStream);
    m_executionStream->flush();
    
    convnet::InputLayer *inputLayer = m_net.getLayer<convnet::InputLayer>("input");
    for (unsigned l = 0; l < inputLayer->getNumOutputs(); l++) {
        convnet::NetworkData &output0 = inputLayer->getOutput(m_states[0], l);
        convnet::NetworkData &output1 = inputLayer->getOutput(m_states[1], l);
        convnet::NetworkData &output2 = inputLayer->getOutput(m_states[2], l);

        convnet::MappedTensor mappedInput0 = dynamic_cast<convnet::TensorData&>(output0).getValues().lock();
        convnet::MappedTensor mappedInput1 = dynamic_cast<convnet::TensorData&>(output1).getValues().lock(true);
        convnet::MappedTensor mappedInput2 = dynamic_cast<convnet::TensorData&>(output2).getValues().lock(true);
        const convnet::TensorSize &inputSize = mappedInput0.getSize();
        for (unsigned i = 0; i < mappedInput0.getNumInstances(); i++) {
            memcpy(mappedInput1.data<float>(i), mappedInput0.data<float>(i), inputSize.numElements() * sizeof(float));
            memcpy(mappedInput2.data<float>(i), mappedInput0.data<float>(i), inputSize.numElements() * sizeof(float));
        }

        if (l == inputIndex) {
            mappedInput1.get<float>(x, y, z, c, n) += epsilon;
            mappedInput2.get<float>(x, y, z, c, n) -= epsilon;
        }

        dynamic_cast<convnet::TensorData&>(output0).getValues().unlock(mappedInput0, false);
        dynamic_cast<convnet::TensorData&>(output1).getValues().unlock(mappedInput1);
        dynamic_cast<convnet::TensorData&>(output2).getValues().unlock(mappedInput2);
    }


    m_net.feedForward(m_states[1], *m_executionWorkspace, *m_executionStream, false);
    m_net.feedForward(m_states[2], *m_executionWorkspace, *m_executionStream, false);
    m_executionStream->flush();

    float finiteGradient, analyticalGradient;

    {

        convnet::MappedTensor mappedOutputValues1 = dynamic_cast<convnet::TensorData&>(*m_states[1].outputs.back()).getValues().lock();
        convnet::MappedTensor mappedOutputValues2 = dynamic_cast<convnet::TensorData&>(*m_states[2].outputs.back()).getValues().lock();
        convnet::MappedTensor mappedOutputGradients0 = dynamic_cast<convnet::TensorData&>(*m_states[0].outputs.back()).getGradients().lock();

        const convnet::TensorSize &outputSize = mappedOutputGradients0.getSize();

        finiteGradient = 0.0f;
        for (unsigned c_ = 0; c_ < outputSize.channels; c_++)
            for (unsigned z_ = 0; z_ < outputSize.depth; z_++)
                for (unsigned y_ = 0; y_ < outputSize.height; y_++)
                    for (unsigned x_ = 0; x_ < outputSize.width; x_++) {
                        float dydx = (mappedOutputValues1.get<float>(x_, y_, z_, c_, n) - mappedOutputValues2.get<float>(x_, y_, z_, c_, n)) / (2.0f * epsilon);

                        finiteGradient += dydx;
                    }

        dynamic_cast<convnet::TensorData&>(*m_states[1].outputs.back()).getValues().unlock(mappedOutputValues1, false);
        dynamic_cast<convnet::TensorData&>(*m_states[2].outputs.back()).getValues().unlock(mappedOutputValues2, false);
        dynamic_cast<convnet::TensorData&>(*m_states[0].outputs.back()).getGradients().unlock(mappedOutputGradients0, false);
    }
    {
        convnet::NetworkData &output0 = inputLayer->getOutput(m_states[0], inputIndex);
        convnet::MappedTensor mappedInputGradients0 = dynamic_cast<convnet::TensorData&>(output0).getGradients().lock();

        analyticalGradient = mappedInputGradients0.get<float>(x, y, z, c, n);

        dynamic_cast<convnet::TensorData&>(output0).getGradients().unlock(mappedInputGradients0, false);
    }

//    std::cout << analyticalGradient << "    " << finiteGradient << std::endl;
    COMPARE_GRADIENTS(finiteGradient, analyticalGradient, 1.0f, 1e-5f);
}

void LossLayerTest::testAll()
{
    convnet::InputLayer *inputLayer = m_net.getLayer<convnet::InputLayer>("input");
    convnet::NetworkData &output0 = inputLayer->getOutput(m_states[0], 0);
    const convnet::TensorSize &inputSize = dynamic_cast<convnet::TensorData&>(output0).getValues().getSize();

    for (unsigned n = 0; n < dynamic_cast<convnet::TensorData&>(output0).getValues().getNumInstances(); n++)
        for (unsigned c = 0; c < inputSize.channels; c++)
            for (unsigned z = 0; z < inputSize.depth; z++)
                for (unsigned y = 0; y < inputSize.height; y++)
                    for (unsigned x = 0; x < inputSize.width; x++)
                        testInput(0, x, y, z, c, n);
}
