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

#include "InputLayerImpl_cudnn.h"


#include "../ConvNetImpl_cudnn.h"
#include "../TensorImpl_cudnn.h"
#include "../ExecutionStreamImpl_cudnn.h"

#include "../CuDnnWrapper.h"

namespace convnet {

InputLayerImpl_cudnn::InputLayerImpl_cudnn(ConvNetImpl_cudnn &convNet, std::string name) :
                    InputLayer(convNet, std::move(name)), m_cudnnWrapper(convNet.getCudnnWrapper())
{
}

void InputLayerImpl_cudnn::resize(ExecutionStream &stream, Key<ConvNet> key)
{
    InputLayer::resize(stream, key);
}

void InputLayerImpl_cudnn::allocateState(NetworkState &networkState, ExecutionStream &/*stream*/, bool forward, bool backward, bool parameterUpdate, Key<ConvNet>) const
{
    for (unsigned i : m_outputs) {
        TensorDataImpl_cudnn *data;
        networkState.outputs[i].reset(data = new TensorDataImpl_cudnn(m_convnet));
        data->allocate(m_connectionList->outputs[i].size, m_convnet.getNumInstances(), DF_FLOAT32, backward);
    }
}

InputStagingBuffer InputLayerImpl_cudnn::allocateStagingBuffer(ExecutionStream &/*stream*/) const
{
    InputStagingBuffer stagingBuffer;
    stagingBuffer.outputs.resize(m_outputs.size());
    for (unsigned i = 0; i < m_outputs.size(); i++) {
        TensorDataImpl_cudnn *data;
        stagingBuffer.outputs[i].reset(data = new TensorDataImpl_cudnn(m_convnet));
        data->allocate(m_connectionList->outputs[m_outputs[i]].size, m_convnet.getNumInstances(), DF_FLOAT32);
    }
    return stagingBuffer;
}


void InputLayerImpl_cudnn::forward(NetworkState &/*networkState*/, ExecutionWorkspace &/*workspace*/, ExecutionStream &/*stream*/, bool /*trainingMode*/) const
{
    /*
    TensorDataImpl_cudnn &input = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[0]]);
    TensorDataImpl_cudnn &output = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_outputs[0]]);
    */
}

void InputLayerImpl_cudnn::backward(NetworkState &/*networkState*/, ExecutionWorkspace &/*workspace*/, ExecutionStream &/*stream*/, bool /*accumulateParameterGradients*/) const
{
    /*
    TensorDataImpl_cudnn &input = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[0]]);
    TensorDataImpl_cudnn &output = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_outputs[0]]);
    */
}

}
