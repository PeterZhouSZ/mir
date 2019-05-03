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

#ifndef POOLINGLAYERIMPL_CUDNN_H
#define POOLINGLAYERIMPL_CUDNN_H

#include "../../layers/PoolingLayer.h"


#include "../cudnn/CuDnnPoolingDescriptor.h"
#include "../cudnn/CuDnnTensorDescriptor.h"

namespace convnet {

class ConvNetImpl_cudnn;
class CuDnnWrapper;

class PoolingLayerImpl_cudnn : public PoolingLayer
{
    public:
        PoolingLayerImpl_cudnn(ConvNetImpl_cudnn &convNet, std::string name);

        virtual void structureChanged() override;

        virtual void resize(ExecutionStream &stream, Key<ConvNet>) override;
        virtual void allocateState(NetworkState &networkState, ExecutionStream &stream, bool forward, bool backward, bool parameterUpdate, Key<ConvNet>) const override;

        virtual void forward(NetworkState &networkState, ExecutionWorkspace &workspace, ExecutionStream &stream, bool trainingMode) const override;
        virtual void backward(NetworkState &networkState, ExecutionWorkspace &workspace, ExecutionStream &stream, bool accumulateParameterGradients) const override;
    protected:
        const CuDnnWrapper &m_cudnnWrapper;

        cudnn::CuDnnPoolingDescriptor m_poolingDesc;
        cudnn::CuDnnTensorDescriptor m_inputTensorDesc;
        cudnn::CuDnnTensorDescriptor m_outputTensorDesc;
};

}

#endif // POOLINGLAYERIMPL_CUDNN_H

