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

#include "ExecutionStreamImpl_cudnn.h"

namespace convnet {

ExecutionStreamWaitingFenceImpl_cudnn::ExecutionStreamWaitingFenceImpl_cudnn() : m_fence(CU_EVENT_DISABLE_TIMING | CU_EVENT_BLOCKING_SYNC)
{
}

void ExecutionStreamWaitingFenceImpl_cudnn::waitFor()
{
    m_fence.waitFor();
}

ExecutionStreamSyncFenceImpl_cudnn::ExecutionStreamSyncFenceImpl_cudnn() : m_fence(CU_EVENT_DISABLE_TIMING)
{
}


ExecutionStreamImpl_cudnn::ExecutionStreamImpl_cudnn() : m_context(m_stream)
{
}

void ExecutionStreamImpl_cudnn::flush()
{
    m_stream.waitFor();
}

void ExecutionStreamImpl_cudnn::insertFence(ExecutionStreamWaitingFence &fence)
{
    ExecutionStreamWaitingFenceImpl_cudnn &f = dynamic_cast<ExecutionStreamWaitingFenceImpl_cudnn&>(fence);
    f.m_fence.insertIntoStream(m_stream);
}

void ExecutionStreamImpl_cudnn::insertFence(ExecutionStreamSyncFence &fence)
{
    ExecutionStreamSyncFenceImpl_cudnn &f = dynamic_cast<ExecutionStreamSyncFenceImpl_cudnn&>(fence);
    f.m_fence.insertIntoStream(m_stream);
}

void ExecutionStreamImpl_cudnn::makeStreamWaitOn(ExecutionStreamSyncFence &fence)
{
    ExecutionStreamSyncFenceImpl_cudnn &f = dynamic_cast<ExecutionStreamSyncFenceImpl_cudnn&>(fence);
    f.m_fence.makeStreamWaitOnFence(m_stream);
}

}
