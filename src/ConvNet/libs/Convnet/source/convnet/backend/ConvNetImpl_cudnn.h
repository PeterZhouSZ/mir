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

#ifndef CONVNETIMPL_CUDNN_H
#define CONVNETIMPL_CUDNN_H

#include "../Convnet.h"
#include "../layers/Layer.h"

#include "CuDnnWrapper.h"

namespace convnet {

class ConvNetImpl_cudnn : public ConvNet
{
    public:
        ConvNetImpl_cudnn(CuDnnWrapper &cudnnWrapper);

        virtual std::unique_ptr<ExecutionWorkspace> allocateExecutionWorkspace(bool forward = true, bool backward = true, bool parameterUpdate = true) override;
        virtual std::unique_ptr<ExecutionStream> allocateExecutionStream() override;

        virtual std::unique_ptr<ExecutionStreamWaitingFence> allocateWaitFence() override;
        virtual std::unique_ptr<ExecutionStreamSyncFence> allocateSyncFence() override;

        inline const CuDnnWrapper &getCudnnWrapper() const { return m_cudnnWrapper; }
        inline CuDnnWrapper &getCudnnWrapper() { return m_cudnnWrapper; }
    protected:
        CuDnnWrapper &m_cudnnWrapper;

        virtual Layer *instantiateLayer(std::string name, const std::string &type) override;
        virtual void restoreLayerFromBlob(FileBlob &/*blob*/) override { throw std::runtime_error("Not implemented!"); }
        virtual void restoreParameterSetFromBlob(FileBlob &/*blob*/) override { throw std::runtime_error("Not implemented!"); }

};


}

#endif // CONVNETIMPL_CUDNN_H
