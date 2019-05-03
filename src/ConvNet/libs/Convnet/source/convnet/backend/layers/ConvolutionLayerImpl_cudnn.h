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

#ifndef CONVOLUTIONLAYERBACKEND_CUDNN_H
#define CONVOLUTIONLAYERBACKEND_CUDNN_H

#include "../../layers/ConvolutionLayer.h"
#include "../TensorImpl_cudnn.h"

#include "../cudnn/CuDnnFilterDescriptor.h"
#include "../cudnn/CuDnnConvolutionDescriptor.h"
#include "../cudnn/CuDnnTensorDescriptor.h"

namespace convnet {

class ConvNetImpl_cudnn;
class CuDnnWrapper;
class ConvolutionLayerImpl_cudnn;

class ConvolutionParametersImpl_cudnn : public ConvolutionParameters
{
    public:
        ConvolutionParametersImpl_cudnn(std::string name, ConvNet &convNet, const CuDnnWrapper &cudnnWrapper);
        virtual ~ConvolutionParametersImpl_cudnn();

        virtual void performParameterStep(float stepsize, ExecutionWorkspace &workspace, ExecutionStream &stream) override;
        virtual void pushToBackend(ExecutionWorkspace &workspace, ExecutionStream &stream) override;
        virtual void pullFromBackend(ExecutionWorkspace &workspace, ExecutionStream &stream) override;

        virtual void restoreSnapshot(ExecutionStream &stream, FileBlob &blob);
        virtual void dumpSnapshot(ExecutionWorkspace &workspace, ExecutionStream &stream, FileBlob &blob);

        virtual void structureChanged() override;

        void printStats();
        
        std::size_t getWorkspaceSize() const;
    protected:

        friend class ConvolutionLayerImpl_cudnn;
        const CuDnnWrapper &m_cudnnWrapper;

        cudnn::CuDnnFilterDescriptor m_filterDesc;
        cudnn::CuDnnTensorDescriptor m_biasDesc;
        TensorImpl_cudnn m_filters;
        TensorImpl_cudnn m_filtersGradient;
        TensorImpl_cudnn m_filtersMomentum;
        TensorImpl_cudnn m_filtersExpectedGradient;
        TensorImpl_cudnn m_bias;
        TensorImpl_cudnn m_biasGradient;
        TensorImpl_cudnn m_biasMomentum;
        TensorImpl_cudnn m_biasExpectedGradient;

        bool m_gradientIsZero = true;
};

class ConvolutionLayerImpl_cudnn : public ConvolutionLayer
{
    public:
        ConvolutionLayerImpl_cudnn(ConvNetImpl_cudnn &convNet, std::string name);

        virtual void resize(ExecutionStream &stream, Key<ConvNet>) override;
        virtual void allocateState(NetworkState &networkState, ExecutionStream &stream, bool forward, bool backward, bool parameterUpdate, Key<ConvNet>) const override;

        virtual void forward(NetworkState &networkState, ExecutionWorkspace &workspace, ExecutionStream &stream, bool trainingMode) const override;
        virtual void backward(NetworkState &networkState, ExecutionWorkspace &workspace, ExecutionStream &stream, bool accumulateParameterGradients) const override;

        virtual std::size_t getWorkspaceSize() override { return m_workspaceMemory; }

        void printStats() { dynamic_cast<ConvolutionParametersImpl_cudnn*>(m_parameters)->printStats(); }
    protected:
        const CuDnnWrapper &m_cudnnWrapper;

        cudnn::CuDnnConvolutionDescriptor m_convDesc;
        cudnn::CuDnnTensorDescriptor m_inputTensorDesc;
        cudnn::CuDnnTensorDescriptor m_outputTensorDesc;

        cudnnConvolutionFwdAlgo_t m_forwardAlgorithm;
        cudnnConvolutionBwdDataAlgo_t m_backwardAlgorithm;
        cudnnConvolutionBwdFilterAlgo_t m_backwardFilterAlgorithm;
        std::size_t m_workspaceMemory;

        virtual ConvolutionParameters *instantiateParameters(std::string name) override;
};

}

#endif // CONVOLUTIONLAYERBACKEND_CUDNN_H
