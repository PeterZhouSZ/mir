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

#include "DropoutLayerImpl_cudnn.h"

#include "../ConvNetImpl_cudnn.h"
#include "../TensorImpl_cudnn.h"
#include "../ExecutionStreamImpl_cudnn.h"
#include "../ExecutionWorkspaceImpl_cudnn.h"

#include "../CuDnnWrapper.h"

#include "../cudnn/CuDnnError.h"

#include "../../Convnet.h"
#include "../cudaKernels/MiscKernel.h"

#include <curand_kernel.h>

#include <iostream>

namespace convnet {

DropoutLayerImpl_cudnn_AuxData::DropoutLayerImpl_cudnn_AuxData(ExecutionStream &stream, const cudnn::CuDnnTensorDescriptor &tensor, float dropout, unsigned seed)
{
    ExecutionStreamImpl_cudnn &streamCuImpl = dynamic_cast<ExecutionStreamImpl_cudnn&>(stream);

    m_rngStateMemory.resize(m_dropoutDesc.getRngStateSize(streamCuImpl.getContext()));
    m_switchesMemory.resize(m_dropoutDesc.getSwitchesSize(streamCuImpl.getContext(), tensor));
    m_dropoutDesc.setup(streamCuImpl.getContext(), dropout, m_rngStateMemory.getPtr(), m_rngStateMemory.size(), seed);
}
    

DropoutLayerImpl_cudnn::DropoutLayerImpl_cudnn(ConvNetImpl_cudnn &convNet, std::string name) :
                    DropoutLayer(convNet, std::move(name)), m_cudnnWrapper(convNet.getCudnnWrapper())
{
}

void DropoutLayerImpl_cudnn::resize(ExecutionStream &stream, Key<ConvNet> key)
{
    DropoutLayer::resize(stream, key);

    const TensorSize &inputSize = m_connectionList->outputs[m_inputs[0]].size;
    //TensorSize &outputSize = m_connectionList->outputs[m_outputs[0]].size;

    m_tensorDesc.setupTightlyPacked({m_convnet.getNumInstances(), inputSize.channels, inputSize.depth, inputSize.height, inputSize.width});

    m_workspaceMemory = inputSize.numElements() * m_convnet.getNumInstances() * sizeof(float);
}

void DropoutLayerImpl_cudnn::allocateState(NetworkState &networkState, ExecutionStream &stream,  bool forward, bool backward, bool parameterUpdate, Key<ConvNet>) const
{
    for (unsigned i : m_outputs) {
        TensorDataImpl_cudnn *data;
        networkState.outputs[i].reset(data = new TensorDataImpl_cudnn(m_convnet));
        data->allocate(m_connectionList->outputs[i].size, m_convnet.getNumInstances(), DF_FLOAT32, backward);
    }
           
    networkState.auxLayerData[m_layerIndex].reset(new DropoutLayerImpl_cudnn_AuxData(
        stream,
        m_tensorDesc,
        m_dropout,
        m_seed
    ));
}

void DropoutLayerImpl_cudnn::forward(NetworkState &networkState, ExecutionWorkspace &/*workspace*/, ExecutionStream &stream, bool trainingMode) const
{
    ExecutionStreamImpl_cudnn &streamCuImpl = dynamic_cast<ExecutionStreamImpl_cudnn&>(stream);
    DropoutLayerImpl_cudnn_AuxData &auxData = dynamic_cast<DropoutLayerImpl_cudnn_AuxData&>(*networkState.auxLayerData[m_layerIndex]);
    auxData.m_trainingMode = trainingMode;
    
    if (auxData.m_dropout != m_dropout) 
		throw std::runtime_error("Dropout changed after allocating state!");

    TensorDataImpl_cudnn &input = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[0]]);    
    TensorDataImpl_cudnn &output = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_outputs[0]]);

    if (trainingMode) {
    
        if (input.getValues().getFormat() != DF_FLOAT32)
            throw std::runtime_error("Only float32 supported!");

        if (output.getValues().getFormat() != DF_FLOAT32)
            throw std::runtime_error("Only float32 supported!");
    
        
        throwCudnnError(cudnnDropoutForward(
            streamCuImpl.getContext().getHandle(),
            auxData.m_dropoutDesc.getDescriptor(),
            m_tensorDesc.getDescriptor(),
            input.getValuesDevicePtr(),
            m_tensorDesc.getDescriptor(),
            output.getValuesDevicePtr(),
            auxData.m_switchesMemory.getPtr(), 
            auxData.m_switchesMemory.size()
        ));
    } else {        
        cuMemcpyAsync((CUdeviceptr)output.getValuesDevicePtr(), 
                      (CUdeviceptr)input.getValuesDevicePtr(), 
                      dynamic_cast<TensorImpl_cudnn&>(input.getValues()).getMemoryAmount(), 
                      streamCuImpl.getStream().getHandle());
    }
}

void DropoutLayerImpl_cudnn::backward(NetworkState &networkState, ExecutionWorkspace &workspace, ExecutionStream &stream, bool /*accumulateParameterGradients*/) const
{
    ExecutionStreamImpl_cudnn &streamCuImpl = dynamic_cast<ExecutionStreamImpl_cudnn&>(stream);
    DropoutLayerImpl_cudnn_AuxData &auxData = dynamic_cast<DropoutLayerImpl_cudnn_AuxData&>(*networkState.auxLayerData[m_layerIndex]);
    ExecutionWorkspaceImpl_cudnn &workspaceCuImpl = dynamic_cast<ExecutionWorkspaceImpl_cudnn&>(workspace);


    TensorDataImpl_cudnn &input = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[0]]);
    TensorDataImpl_cudnn &output = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_outputs[0]]);

    if (input.getGradients().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");

    if (output.getGradients().getFormat() != DF_FLOAT32)
        throw std::runtime_error("Only float32 supported!");

#if 0
    throwCudnnError(cudnnDropoutBackward(
        streamCuImpl.getContext().getHandle(),
        auxData.m_dropoutDesc.getDescriptor(),
        m_tensorDesc.getDescriptor(),
        output.getGradientsDevicePtr(),
        m_tensorDesc.getDescriptor(),
        input.getGradientsDevicePtr(),
        auxData.m_switchesMemory.getPtr(), 
        auxData.m_switchesMemory.size()
    ));
#else
    if (auxData.m_trainingMode) {
        if (workspaceCuImpl.getWorkspaceMemory().size() < dynamic_cast<TensorImpl_cudnn&>(input.getGradients()).getMemoryAmount())
            throw std::runtime_error("Insufficient workspace memory. internal error!");
        
        throwCudnnError(cudnnDropoutBackward(
            streamCuImpl.getContext().getHandle(),
            auxData.m_dropoutDesc.getDescriptor(),
            m_tensorDesc.getDescriptor(),
            output.getGradientsDevicePtr(),
            m_tensorDesc.getDescriptor(),
            workspaceCuImpl.getWorkspaceMemory().getPtr(),
            auxData.m_switchesMemory.getPtr(),
            auxData.m_switchesMemory.size()
        ));

        const CudaUtils::CudaKernel &kernel = m_cudnnWrapper.getAuxKernels().getInplaceSumKernel();

        InplaceSumParams params;
        params.dst = (float*) input.getGradientsDevicePtr();
        params.src = (float*) workspaceCuImpl.getWorkspaceMemory().getPtr();
        params.numElements = input.getGradients().getTotalNumberOfElements();

        kernel.launch({256u, 1u, 1u},
                        {(params.numElements + 256u-1u)/256u, 1, 1},
                    &params, sizeof(params),
                    streamCuImpl.getStream());
    } else {
        cuMemcpyAsync((CUdeviceptr)input.getGradientsDevicePtr(), 
                      (CUdeviceptr)output.getGradientsDevicePtr(), 
                      dynamic_cast<TensorImpl_cudnn&>(input.getGradients()).getMemoryAmount(), 
                      streamCuImpl.getStream().getHandle());
    }
#endif
}

}
