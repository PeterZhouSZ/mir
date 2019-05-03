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

#ifndef INPUTLAYERIMPL_CUDNN_H
#define INPUTLAYERIMPL_CUDNN_H

#include "../../layers/InputLayer.h"

namespace convnet {

class ConvNetImpl_cudnn;
class CuDnnWrapper;

class InputLayerImpl_cudnn : public InputLayer
{
    public:
        InputLayerImpl_cudnn(ConvNetImpl_cudnn &convNet, std::string name);

        virtual void resize(ExecutionStream &stream, Key<ConvNet>) override;
        virtual void allocateState(NetworkState &networkState, ExecutionStream &stream, bool forward, bool backward, bool parameterUpdate, Key<ConvNet>) const override;
        virtual InputStagingBuffer allocateStagingBuffer(ExecutionStream &stream) const override;


        virtual void forward(NetworkState &networkState, ExecutionWorkspace &workspace, ExecutionStream &stream, bool trainingMode) const override;
        virtual void backward(NetworkState &networkState, ExecutionWorkspace &workspace, ExecutionStream &stream, bool accumulateParameterGradients) const override;
    protected:
        const CuDnnWrapper &m_cudnnWrapper;
};

}

#endif // INPUTLAYERIMPL_CUDNN_H
