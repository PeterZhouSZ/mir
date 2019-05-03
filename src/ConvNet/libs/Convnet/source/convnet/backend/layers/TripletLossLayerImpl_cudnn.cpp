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

#include "TripletLossLayerImpl_cudnn.h"

#include "../ConvNetImpl_cudnn.h"
#include "../TensorImpl_cudnn.h"
#include "../ExecutionStreamImpl_cudnn.h"
#include "../cudaKernels/TripletLossKernel.h"

#include "../CuDnnWrapper.h"

#include "../cudnn/CuDnnError.h"

#include "../../Convnet.h"

namespace convnet {

TripletLossLayerImpl_cudnn::TripletLossLayerImpl_cudnn(ConvNetImpl_cudnn &convNet, std::string name) :
                    TripletLossLayer(convNet, std::move(name)), m_cudnnWrapper(convNet.getCudnnWrapper())
{
}

void TripletLossLayerImpl_cudnn::resize(ExecutionStream &stream, Key<ConvNet> key)
{
    TripletLossLayer::resize(stream, key);
}

void TripletLossLayerImpl_cudnn::allocateState(NetworkState &networkState, ExecutionStream &/*stream*/, bool /*forward*/, bool backward, bool /*parameterUpdate*/, Key<ConvNet>) const
{
    for (unsigned i : m_outputs) {
        TensorDataImpl_cudnn *data;
        networkState.outputs[i].reset(data = new TensorDataImpl_cudnn(m_convnet));
        data->allocate(m_connectionList->outputs[i].size, m_convnet.getNumInstances(), DF_FLOAT32, backward);
    }
}

void TripletLossLayerImpl_cudnn::forward(NetworkState &networkState, ExecutionWorkspace &/*workspace*/, ExecutionStream &stream, bool /*trainingMode*/) const
{
    ExecutionStreamImpl_cudnn &streamCuImpl = dynamic_cast<ExecutionStreamImpl_cudnn&>(stream);


    TensorDataImpl_cudnn &input0 = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[0]]);
    TensorDataImpl_cudnn &input1 = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[1]]);
    TensorDataImpl_cudnn &input2 = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[2]]);
    TensorDataImpl_cudnn &output = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_outputs[0]]);

    if (input0.getValues().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");

    if (input1.getValues().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");

    if (input2.getValues().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");

    if (output.getValues().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");

    const CudaUtils::CudaKernel &kernel = m_cudnnWrapper.getAuxKernels().getTripletLossForwardKernel();

    unsigned width = input0.getValues().getSize().width;
    unsigned height = input0.getValues().getSize().height;
    unsigned depth = input0.getValues().getSize().depth;
    unsigned channels = input0.getValues().getSize().channels;

    TripletLossForwardParams params;
    
    params.inputAnchor = (float*) input0.getValuesDevicePtr();
    params.inputMatch = (float*) input1.getValuesDevicePtr();
    params.inputNonMatch = (float*) input2.getValuesDevicePtr();
    params.loss = (float*) output.getValuesDevicePtr();
    params.weight = m_weight;
    params.margin = m_margin;

    params.widthHeightDepth = width*height*depth;
    params.channels = channels;
    params.strideN = width*height*depth*channels;

    kernel.launch({32u, 1u, 1u},
                {(width*height*depth + 32u-1u)/32u, m_convnet.getNumInstances(), 1u},
                &params, sizeof(params),
                streamCuImpl.getStream(),
                0
                );

}

void TripletLossLayerImpl_cudnn::backward(NetworkState &networkState, ExecutionWorkspace &/*workspace*/, ExecutionStream &stream, bool /*accumulateParameterGradients*/) const
{
    if (m_evalOnly) return;

    ExecutionStreamImpl_cudnn &streamCuImpl = dynamic_cast<ExecutionStreamImpl_cudnn&>(stream);


    TensorDataImpl_cudnn &input0 = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[0]]);
    TensorDataImpl_cudnn &input1 = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[1]]);
    TensorDataImpl_cudnn &input2 = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[2]]);
    TensorDataImpl_cudnn &output = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_outputs[0]]);

    if (input0.getValues().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");

    if (input1.getValues().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");

    if (input2.getValues().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");

    if (output.getValues().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");

    const CudaUtils::CudaKernel &kernel = m_cudnnWrapper.getAuxKernels().getTripletLossBackwardKernel();

    unsigned width = input0.getValues().getSize().width;
    unsigned height = input0.getValues().getSize().height;
    unsigned depth = input0.getValues().getSize().depth;
    unsigned channels = input0.getValues().getSize().channels;

    TripletLossBackwardParams params;
    
    params.inputAnchor = (float*) input0.getValuesDevicePtr();
    params.inputMatch = (float*) input1.getValuesDevicePtr();
    params.inputNonMatch = (float*) input2.getValuesDevicePtr();
    params.gradientAnchor = (float*) input0.getGradientsDevicePtr();
    params.gradientMatch = (float*) input1.getGradientsDevicePtr();
    params.gradientNonMatch = (float*) input2.getGradientsDevicePtr();
    params.loss = (float*) output.getValuesDevicePtr();
    params.weight = m_weight;
    params.margin = m_margin;

    params.widthHeightDepth = width*height*depth;
    params.channels = channels;
    params.strideN = width*height*depth*channels;

    kernel.launch({32u, 1u, 1u},
                {(width*height*depth + 32u-1u)/32u, m_convnet.getNumInstances(), 1u},
                &params, sizeof(params),
                streamCuImpl.getStream(),
                0
                );

}

}
