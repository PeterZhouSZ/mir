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

#ifndef EXECUTIONSTREAMIMPL_CUDNN_H
#define EXECUTIONSTREAMIMPL_CUDNN_H

#include "../ExecutionStream.h"
#include "cudnn/CuDnnContext.h"

#include <cudaUtils/CudaFence.h>
#include <cudaUtils/CudaStream.h>


namespace convnet {

class ExecutionStreamImpl_cudnn;

class ExecutionStreamWaitingFenceImpl_cudnn : public ExecutionStreamWaitingFence
{
    public:
        ExecutionStreamWaitingFenceImpl_cudnn();

        virtual void waitFor() override;
    protected:
        friend class ExecutionStreamImpl_cudnn;
        CudaUtils::CudaFence m_fence;
};

class ExecutionStreamSyncFenceImpl_cudnn : public ExecutionStreamSyncFence
{
    public:
        ExecutionStreamSyncFenceImpl_cudnn();
    protected:
        friend class ExecutionStreamImpl_cudnn;
        CudaUtils::CudaFence m_fence;
};


class ExecutionStreamImpl_cudnn : public ExecutionStream
{
    public:
        ExecutionStreamImpl_cudnn();

        virtual void flush() override;

        virtual void insertFence(ExecutionStreamWaitingFence &fence) override;
        virtual void insertFence(ExecutionStreamSyncFence &fence) override;

        virtual void makeStreamWaitOn(ExecutionStreamSyncFence &fence) override;

        inline CudaUtils::CudaStream &getStream() { return m_stream; }
        inline cudnn::CuDnnContext &getContext() { return m_context; }
    protected:
        CudaUtils::CudaStream m_stream;
        cudnn::CuDnnContext m_context;
};

}

#endif // EXECUTIONSTREAMIMPL_CUDNN_H
