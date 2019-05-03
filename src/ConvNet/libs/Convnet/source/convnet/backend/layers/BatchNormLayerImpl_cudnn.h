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

#ifndef BATCHNORMLAYERIMPL_CUDNN_H
#define BATCHNORMLAYERIMPL_CUDNN_H

#include "../../layers/BatchNormLayer.h"
#include "../TensorImpl_cudnn.h"

#include "../cudnn/CuDnnFilterDescriptor.h"
#include "../cudnn/CuDnnConvolutionDescriptor.h"
#include "../cudnn/CuDnnTensorDescriptor.h"

namespace convnet {

class ConvNetImpl_cudnn;
class CuDnnWrapper;
class ConvolutionLayerImpl_cudnn;

class BatchNormParametersImpl_cudnn : public BatchNormParameters
{
    public:
        BatchNormParametersImpl_cudnn(std::string name, ConvNet &convNet, const CuDnnWrapper &cudnnWrapper);
        virtual ~BatchNormParametersImpl_cudnn();

        virtual void performParameterStep(float stepsize, ExecutionWorkspace &workspace, ExecutionStream &stream) override;
        virtual void pushToBackend(ExecutionWorkspace &workspace, ExecutionStream &stream) override;
        virtual void pullFromBackend(ExecutionWorkspace &workspace, ExecutionStream &stream) override;

        
        virtual float computeRegularizationLoss(ExecutionWorkspace &/*workspace*/, ExecutionStream &/*stream*/) { return 0.0f; }
        
        virtual void restoreSnapshot(ExecutionStream &/*stream*/, FileBlob &/*blob*/) { }
        virtual void dumpSnapshot(ExecutionWorkspace &/*workspace*/, ExecutionStream &/*stream*/, FileBlob &/*blob*/) { }
        /*
        virtual void restoreSnapshot(ExecutionStream &stream, FileBlob &blob);
        virtual void dumpSnapshot(ExecutionWorkspace &workspace, ExecutionStream &stream, FileBlob &blob);
*/
        virtual void structureChanged() override;
    protected:
        friend class BatchNormLayerImpl_cudnn;
        const CuDnnWrapper &m_cudnnWrapper;

        cudnn::CuDnnTensorDescriptor m_tensorDesc;

        TensorImpl_cudnn m_runningMean;
        TensorImpl_cudnn m_runningVar;

        TensorImpl_cudnn m_scale;
        TensorImpl_cudnn m_scaleGradient;
        TensorImpl_cudnn m_scaleMomentum;
        TensorImpl_cudnn m_scaleExpectedGradient;

        TensorImpl_cudnn m_bias;
        TensorImpl_cudnn m_biasGradient;
        TensorImpl_cudnn m_biasMomentum;
        TensorImpl_cudnn m_biasExpectedGradient; 
};

class BatchNormLayerImpl_cudnn_AuxData : public AuxiliaryNetworkLayerData
{
    public:
        BatchNormLayerImpl_cudnn_AuxData(ConvNet &convNet, unsigned numChannels);
        
        TensorImpl_cudnn m_mean;
        TensorImpl_cudnn m_invVar;
        bool m_trainingMode;
};


class BatchNormLayerImpl_cudnn : public BatchNormLayer
{
    public:
        BatchNormLayerImpl_cudnn(ConvNetImpl_cudnn &convNet, std::string name);

        virtual void resize(ExecutionStream &stream, Key<ConvNet>) override;
        virtual void allocateState(NetworkState &networkState, ExecutionStream &stream, bool forward, bool backward, bool parameterUpdate, Key<ConvNet>) const override;

        virtual void forward(NetworkState &networkState, ExecutionWorkspace &workspace, ExecutionStream &stream, bool trainingMode) const override;
        virtual void backward(NetworkState &networkState, ExecutionWorkspace &workspace, ExecutionStream &stream, bool accumulateParameterGradients) const override;

        inline std::size_t getCurrentWorkspaceMemoryDemands() const { return m_workspaceMemory; }
    protected:
        const CuDnnWrapper &m_cudnnWrapper;

        cudnn::CuDnnTensorDescriptor m_inputTensorDesc;
        cudnn::CuDnnTensorDescriptor m_outputTensorDesc;
        

        std::size_t m_workspaceMemory;

        virtual BatchNormParameters *instantiateParameters(std::string name) override;
};

}


#endif // BATCHNORMLAYERIMPL_CUDNN_H
