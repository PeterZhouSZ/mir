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

#ifndef DROPOUTLAYERIMPL_CUDNN_H
#define DROPOUTLAYERIMPL_CUDNN_H

#include "../../layers/DropoutLayer.h"

#include "../cudnn/CuDnnTensorDescriptor.h"
#include "../cudnn/CuDnnDropoutDescriptor.h"

#include <cudaUtils/CudaDeviceMemory.h>

namespace convnet {

class ConvNetImpl_cudnn;
class CuDnnWrapper;

class DropoutLayerImpl_cudnn_AuxData : public AuxiliaryNetworkLayerData
{
    public:
        DropoutLayerImpl_cudnn_AuxData(ExecutionStream &stream, const cudnn::CuDnnTensorDescriptor &tensor, float dropout, unsigned seed);
        
        CudaUtils::CudaDeviceMemory m_switchesMemory;
        CudaUtils::CudaDeviceMemory m_rngStateMemory;
        cudnn::CuDnnDropoutDescriptor m_dropoutDesc;
        bool m_trainingMode;
        float m_dropout;
};

class DropoutLayerImpl_cudnn : public DropoutLayer
{
    public:
        DropoutLayerImpl_cudnn(ConvNetImpl_cudnn &convNet, std::string name);
        
        virtual void resize(ExecutionStream &stream, Key<ConvNet>) override;
        virtual void allocateState(NetworkState &networkState, ExecutionStream &stream, bool forward, bool backward, bool parameterUpdate, Key<ConvNet>) const override;
        
        virtual void forward(NetworkState &networkState, ExecutionWorkspace &workspace, ExecutionStream &stream, bool trainingMode) const override;
        virtual void backward(NetworkState &networkState, ExecutionWorkspace &workspace, ExecutionStream &stream, bool accumulateParameterGradients) const override;
        
        virtual std::size_t getWorkspaceSize() override { return m_workspaceMemory; }
    protected:
        const CuDnnWrapper &m_cudnnWrapper;
        
        cudnn::CuDnnTensorDescriptor m_tensorDesc;
        std::size_t m_workspaceMemory;

};

}


#endif // DROPOUTLAYERIMPL_CUDNN_H
