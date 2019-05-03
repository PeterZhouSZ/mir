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

#include "TensorImpl_cudnn.h"

#include "ExecutionStreamImpl_cudnn.h"

#include <cudaUtils/CudaDriver.h>

#include "ConvNetImpl_cudnn.h"

namespace convnet {

void TensorImpl_cudnn::allocate(const TensorSize &size, unsigned numInstances, DataFormat format)
{
    m_numInstances = numInstances;
    m_size = size;
    m_format = format;
    m_deviceMemory.resize(getMemoryRequirements());
}

MappedTensor TensorImpl_cudnn::lock(bool discard)
{
    ConvNetImpl_cudnn &convnet = dynamic_cast<ConvNetImpl_cudnn &>(m_convnet);

    void *ptr = convnet.getCudnnWrapper().getPinnedMemoryAllocator().allocate(m_deviceMemory.size(), discard);

    MappedTensor mappedTensor(ptr, m_size, m_numInstances, m_format);
    if (!discard)
        m_deviceMemory.downloadSync(mappedTensor.data<char>(0), m_deviceMemory.size(), 0);

    return mappedTensor;
}

void TensorImpl_cudnn::unlock(MappedTensor mappedTensor, bool dirty)
{
    if (dirty)
        m_deviceMemory.uploadSync(mappedTensor.data<char>(0), m_deviceMemory.size(), 0);

    ConvNetImpl_cudnn &convnet = dynamic_cast<ConvNetImpl_cudnn &>(m_convnet);
    convnet.getCudnnWrapper().getPinnedMemoryAllocator().free(mappedTensor.data<char>(0));
}


void TensorImpl_cudnn::setZero(ExecutionStream &stream)
{
    ExecutionStreamImpl_cudnn &streamCuImpl = dynamic_cast<ExecutionStreamImpl_cudnn&>(stream);

    if (m_deviceMemory.size() % 4 == 0)
        cuMemsetD32Async((CUdeviceptr) m_deviceMemory.getPtr(), 0, m_deviceMemory.size()/4, streamCuImpl.getStream().getHandle());
    else
        cuMemsetD8Async((CUdeviceptr) m_deviceMemory.getPtr(), 0, m_deviceMemory.size(), streamCuImpl.getStream().getHandle());
}

void TensorImpl_cudnn::setZeroSync()
{
    if (m_deviceMemory.size() % 4 == 0)
        cuMemsetD32((CUdeviceptr) m_deviceMemory.getPtr(), 0, m_deviceMemory.size()/4);
    else
        cuMemsetD8((CUdeviceptr) m_deviceMemory.getPtr(), 0, m_deviceMemory.size());
}


}
