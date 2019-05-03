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

#include "PoolingLayerImpl_cudnn.h"

#include "../ConvNetImpl_cudnn.h"
#include "../TensorImpl_cudnn.h"
#include "../ExecutionStreamImpl_cudnn.h"


#include "../CuDnnWrapper.h"

#include "../cudnn/CuDnnError.h"

#include "../../Convnet.h"

namespace convnet {

PoolingLayerImpl_cudnn::PoolingLayerImpl_cudnn(ConvNetImpl_cudnn &convNet, std::string name) :
                    PoolingLayer(convNet, std::move(name)), m_cudnnWrapper(convNet.getCudnnWrapper())
{
    structureChanged();
}

void PoolingLayerImpl_cudnn::structureChanged()
{
    m_poolingDesc.setup({m_windowSize[2], m_windowSize[1], m_windowSize[0]},
                        {0, 0, 0},
                        {m_windowSize[2], m_windowSize[1], m_windowSize[0]},
                        m_mode == MODE_AVG?CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING:CUDNN_POOLING_MAX);
}


void PoolingLayerImpl_cudnn::resize(ExecutionStream &stream, Key<ConvNet> key)
{
    PoolingLayer::resize(stream, key);

    const TensorSize &inputSize = m_connectionList->outputs[m_inputs[0]].size;
    TensorSize &outputSize = m_connectionList->outputs[m_outputs[0]].size;

    m_inputTensorDesc.setupTightlyPacked({m_convnet.getNumInstances(), inputSize.channels, inputSize.depth, inputSize.height, inputSize.width});
    m_outputTensorDesc.setupTightlyPacked({m_convnet.getNumInstances(), outputSize.channels, outputSize.depth, outputSize.height, outputSize.width});
/*
    int dims[5];
    throwCudnnError(cudnnGetPoolingNdForwardOutputDim(
        m_poolingDesc.getDescriptor(),
        m_inputTensorDesc.getDescriptor(),
        5,
        dims));

    std::cout << "Pooling input: " << inputSize << std::endl;
    std::cout << "Pooling output: " << outputSize << std::endl;
    std::cout << "cudnn: " << dims[0] << "x" << dims[1] << "x" << dims[2] << "x" << dims[3] << "x" << dims[4] << std::endl;
*/
}

void PoolingLayerImpl_cudnn::allocateState(NetworkState &networkState, ExecutionStream &/*stream*/, bool forward, bool backward, bool parameterUpdate, Key<ConvNet>) const
{
    for (unsigned i : m_outputs) {
        TensorDataImpl_cudnn *data;
        networkState.outputs[i].reset(data = new TensorDataImpl_cudnn(m_convnet));
        data->allocate(m_connectionList->outputs[i].size, m_convnet.getNumInstances(), DF_FLOAT32, backward);
    }
}

void PoolingLayerImpl_cudnn::forward(NetworkState &networkState, ExecutionWorkspace &/*workspace*/, ExecutionStream &stream, bool /*trainingMode*/) const
{
    ExecutionStreamImpl_cudnn &streamCuImpl = dynamic_cast<ExecutionStreamImpl_cudnn&>(stream);


    TensorDataImpl_cudnn &input = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[0]]);
    TensorDataImpl_cudnn &output = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_outputs[0]]);

    if (input.getValues().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");

    if (output.getValues().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");

    float one = 1.0f;
    float zero = 0.0f;

    throwCudnnError(cudnnPoolingForward(
        streamCuImpl.getContext().getHandle(),
        m_poolingDesc.getDescriptor(),
        &one,
        m_inputTensorDesc.getDescriptor(),
        input.getValuesDevicePtr(),
        &zero,
        m_outputTensorDesc.getDescriptor(),
        output.getValuesDevicePtr()
    ));
}

void PoolingLayerImpl_cudnn::backward(NetworkState &networkState, ExecutionWorkspace &/*workspace*/, ExecutionStream &stream, bool /*accumulateParameterGradients*/) const
{
    ExecutionStreamImpl_cudnn &streamCuImpl = dynamic_cast<ExecutionStreamImpl_cudnn&>(stream);

    TensorDataImpl_cudnn &input = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[0]]);
    TensorDataImpl_cudnn &output = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_outputs[0]]);

    if (input.getValues().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");

    if (input.getGradients().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");

    if (output.getGradients().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");

    float one = 1.0f;
    //float zero = 0.0f;

    throwCudnnError(cudnnPoolingBackward(
        streamCuImpl.getContext().getHandle(),
        m_poolingDesc.getDescriptor(),
        &one,
        m_outputTensorDesc.getDescriptor(),
        output.getValuesDevicePtr(),
        m_outputTensorDesc.getDescriptor(),
        output.getGradientsDevicePtr(),
        m_inputTensorDesc.getDescriptor(),
        input.getValuesDevicePtr(),
        &one,
        m_inputTensorDesc.getDescriptor(),
        input.getGradientsDevicePtr()
    ));
}

}
