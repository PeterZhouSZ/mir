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

#include "CrossEntropyLossLayerImpl_cudnn.h"

#include "../ConvNetImpl_cudnn.h"
#include "../TensorImpl_cudnn.h"
#include "../ExecutionStreamImpl_cudnn.h"
#include "../cudaKernels/CrossEntropyLossKernel.h"

#include "../CuDnnWrapper.h"

#include "../cudnn/CuDnnError.h"

#include "../../Convnet.h"

#include <iostream>

namespace convnet {

CrossEntropyLossLayerImpl_cudnn::CrossEntropyLossLayerImpl_cudnn(ConvNetImpl_cudnn &convNet, std::string name) :
                    CrossEntropyLossLayer(convNet, std::move(name)), m_cudnnWrapper(convNet.getCudnnWrapper())
{
}

void CrossEntropyLossLayerImpl_cudnn::resize(ExecutionStream &stream, Key<ConvNet> key)
{
    CrossEntropyLossLayer::resize(stream, key);
    std::cout << "Cross entropy output size is " << m_connectionList->outputs[m_outputs[0]].size << std::endl;
}

void CrossEntropyLossLayerImpl_cudnn::allocateState(NetworkState &networkState, ExecutionStream &/*stream*/, bool forward, bool backward, bool parameterUpdate, Key<ConvNet>) const
{
    for (unsigned i : m_outputs) {
        TensorDataImpl_cudnn *data;
        networkState.outputs[i].reset(data = new TensorDataImpl_cudnn(m_convnet));
        data->allocate(m_connectionList->outputs[i].size, m_convnet.getNumInstances(), DF_FLOAT32, backward);
    }
}

void CrossEntropyLossLayerImpl_cudnn::forward(NetworkState &networkState, ExecutionWorkspace &/*workspace*/, ExecutionStream &stream, bool /*trainingMode*/) const
{
    ExecutionStreamImpl_cudnn &streamCuImpl = dynamic_cast<ExecutionStreamImpl_cudnn&>(stream);


    TensorDataImpl_cudnn &input0 = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[0]]);
    TensorDataImpl_cudnn &input1 = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[1]]);
    TensorDataImpl_cudnn &output = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_outputs[0]]);

    if (input0.getValues().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");

    if (input1.getValues().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");

    if (output.getValues().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");


    if ((input0.getValues().getSize().width == 1) && 
        (input0.getValues().getSize().height == 1) &&
        (input0.getValues().getSize().depth == 1) &&
        (input1.getValues().getSize().width == 1) && 
        (input1.getValues().getSize().height == 1) &&
        (input1.getValues().getSize().depth == 1)) {
        
        const CudaUtils::CudaKernel &kernel = m_cudnnWrapper.getAuxKernels().getCrossEntropyLossForwardKernel(true);

        unsigned sharedMem = ScalarCrossEntropyLoss_forwardSharedMemorySize(input0.getValues().getSize().channels, input0.getValues().getNumInstances());


        ScalarCrossEntropyLossForwardParams params;
        params.inputValues = (float*) input0.getValuesDevicePtr();
        params.inputLabels = (float*) input1.getValuesDevicePtr();
        params.loss = (float*) output.getValuesDevicePtr();
        params.numLabels = input0.getValues().getSize().channels;
        params.scale = m_weight;


        kernel.launch({input0.getValues().getNumInstances(), 1u, 1u},
                    {1u, 1u, 1u},
                    &params, sizeof(params),
                    streamCuImpl.getStream(),
                    sharedMem
                    );
    } else {
        const CudaUtils::CudaKernel &kernel = m_cudnnWrapper.getAuxKernels().getCrossEntropyLossForwardKernel(false);
        
        unsigned width = output.getValues().getSize().width;
        unsigned height = output.getValues().getSize().height;
        unsigned depth = output.getValues().getSize().depth;


        unsigned sharedMem = CrossEntropyLoss_forwardSharedMemorySize(input0.getValues().getSize().channels, 32u);

        unsigned offsetX_i0 = (input0.getValues().getSize().width - width)/2;
        unsigned offsetY_i0 = (input0.getValues().getSize().height - height)/2;
        unsigned offsetZ_i0 = (input0.getValues().getSize().depth - depth)/2;

        unsigned offsetX_i1 = (input1.getValues().getSize().width - width)/2;
        unsigned offsetY_i1 = (input1.getValues().getSize().height - height)/2;
        unsigned offsetZ_i1 = (input1.getValues().getSize().depth - depth)/2;


        CrossEntropyLossForwardParams params;
        params.strideX[0] = 1;
        params.strideY[0] = params.strideX[0] * input0.getValues().getSize().width;
        params.strideZ[0] = params.strideY[0] * input0.getValues().getSize().height;
        params.strideC[0] = params.strideZ[0] * input0.getValues().getSize().depth;
        params.strideN[0] = params.strideC[0] * input0.getValues().getSize().channels;

        params.strideX[1] = 1;
        params.strideY[1] = params.strideX[1] * input1.getValues().getSize().width;
        params.strideZ[1] = params.strideY[1] * input1.getValues().getSize().height;
        params.strideN[1] = params.strideZ[1] * input1.getValues().getSize().depth;

        params.strideX[2] = 1;
        params.strideY[2] = params.strideX[2] * output.getValues().getSize().width;
        params.strideZ[2] = params.strideY[2] * output.getValues().getSize().height;
        params.strideN[2] = params.strideZ[2] * output.getValues().getSize().depth;


        params.inputInferredLabels = (float*) input0.getValuesDevicePtr()
                                        + offsetX_i0 * params.strideX[0]
                                        + offsetY_i0 * params.strideY[0]
                                        + offsetZ_i0 * params.strideZ[0];

        params.inputGTLabels = (float*) input1.getValuesDevicePtr()
                                        + offsetX_i1 * params.strideX[1]
                                        + offsetY_i1 * params.strideY[1]
                                        + offsetZ_i1 * params.strideZ[1];
        params.loss = (float*) output.getValuesDevicePtr();
        params.numInstances = output.getValues().getNumInstances();
        params.numLabels = input0.getValues().getSize().channels;

        params.width = width;
        params.height = height;
        params.depth = depth;
        params.scale = m_weight;

        kernel.launch({32u, 1u, 1u},
                    {(width*height*depth*params.numInstances + 32u-1u)/32u, 1u, 1u},
                    &params, sizeof(params),
                    streamCuImpl.getStream(),
                    sharedMem
                    );
    }
}

void CrossEntropyLossLayerImpl_cudnn::backward(NetworkState &networkState, ExecutionWorkspace &/*workspace*/, ExecutionStream &stream, bool /*accumulateParameterGradients*/) const
{
    ExecutionStreamImpl_cudnn &streamCuImpl = dynamic_cast<ExecutionStreamImpl_cudnn&>(stream);


    TensorDataImpl_cudnn &input0 = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[0]]);
    TensorDataImpl_cudnn &input1 = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[1]]);
    TensorDataImpl_cudnn &output = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_outputs[0]]);

    if (input0.getValues().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");

    if (input1.getValues().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");
/*
    if (output.getValues().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");
*/

    if ((input0.getValues().getSize().width == 1) && 
        (input0.getValues().getSize().height == 1) &&
        (input0.getValues().getSize().depth == 1) &&
        (input1.getValues().getSize().width == 1) && 
        (input1.getValues().getSize().height == 1) &&
        (input1.getValues().getSize().depth == 1)) {


        const CudaUtils::CudaKernel &kernel = m_cudnnWrapper.getAuxKernels().getCrossEntropyLossBackwardKernel(true);

        unsigned sharedMem = ScalarCrossEntropyLoss_backwardSharedMemorySize(input0.getValues().getSize().channels, input0.getValues().getNumInstances());


        ScalarCrossEntropyLossBackwardParams params;
        params.inputValues = (float*) input0.getValuesDevicePtr();
        params.inputLabels = (float*) input1.getValuesDevicePtr();
        params.gradients = (float*) input0.getGradientsDevicePtr();
        params.numLabels = input0.getValues().getSize().channels;

        params.scale = m_weight;

        kernel.launch({input0.getValues().getNumInstances(), 1u, 1u},
                    {1u, 1u, 1u},
                    &params, sizeof(params),
                    streamCuImpl.getStream(),
                    sharedMem
                    );
    } else {
        const CudaUtils::CudaKernel &kernel = m_cudnnWrapper.getAuxKernels().getCrossEntropyLossBackwardKernel(false);

        unsigned width = output.getValues().getSize().width;
        unsigned height = output.getValues().getSize().height;
        unsigned depth = output.getValues().getSize().depth;


        unsigned sharedMem = CrossEntropyLoss_forwardSharedMemorySize(input0.getValues().getSize().channels, 32u);

        unsigned offsetX_i0 = (input0.getValues().getSize().width - width)/2;
        unsigned offsetY_i0 = (input0.getValues().getSize().height - height)/2;
        unsigned offsetZ_i0 = (input0.getValues().getSize().depth - depth)/2;

        unsigned offsetX_i1 = (input1.getValues().getSize().width - width)/2;
        unsigned offsetY_i1 = (input1.getValues().getSize().height - height)/2;
        unsigned offsetZ_i1 = (input1.getValues().getSize().depth - depth)/2;

        CrossEntropyLossBackwardParams params;
        params.strideX[0] = 1;
        params.strideY[0] = params.strideX[0] * input0.getValues().getSize().width;
        params.strideZ[0] = params.strideY[0] * input0.getValues().getSize().height;
        params.strideC[0] = params.strideZ[0] * input0.getValues().getSize().depth;
        params.strideN[0] = params.strideC[0] * input0.getValues().getSize().channels;

        params.strideX[1] = 1;
        params.strideY[1] = params.strideX[1] * input1.getValues().getSize().width;
        params.strideZ[1] = params.strideY[1] * input1.getValues().getSize().height;
        params.strideN[1] = params.strideZ[1] * input1.getValues().getSize().depth;

        params.inputInferredLabels = (float*) input0.getValuesDevicePtr()
                                        + offsetX_i0 * params.strideX[0]
                                        + offsetY_i0 * params.strideY[0]
                                        + offsetZ_i0 * params.strideZ[0];

        params.inputGTLabels = (float*) input1.getValuesDevicePtr()
                                        + offsetX_i1 * params.strideX[1]
                                        + offsetY_i1 * params.strideY[1]
                                        + offsetZ_i1 * params.strideZ[1];
        params.gradients = (float*) input0.getGradientsDevicePtr();
        params.numInstances = input0.getValues().getNumInstances();
        params.numLabels = input0.getValues().getSize().channels;

        params.width = width;
        params.height = height;
        params.depth = depth;
        params.scale = m_weight;

        kernel.launch({32u, 1u, 1u},
                    {(width*height*depth*params.numInstances + 32u-1u)/32u, 1u, 1u},
                    &params, sizeof(params),
                    streamCuImpl.getStream(),
                    sharedMem
                    );
    }
}

}
