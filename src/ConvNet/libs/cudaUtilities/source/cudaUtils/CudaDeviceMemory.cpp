/*
 * Cuda Utilities - Distributed for "Mental Image Retrieval" implementation
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

#include "CudaDeviceMemory.h"
#include "CudaDriver.h"
#include <cuda.h>
#include <stdexcept>

namespace CudaUtils {


void CudaBaseDeviceMemory::uploadAsync(const void *src, size_t size, size_t offset, const CudaStream &stream)
{
    CudaDriver::throwOnCudaError(cuMemcpyHtoDAsync((CUdeviceptr)m_ptr + offset, src, size, stream.getHandle()), __FILE__, __LINE__);
}

void CudaBaseDeviceMemory::downloadAsync(void *dst, size_t size, size_t offset, const CudaStream &stream) const
{
    CudaDriver::throwOnCudaError(cuMemcpyDtoHAsync(dst, (CUdeviceptr)m_ptr + offset, size, stream.getHandle()), __FILE__, __LINE__);
}

void CudaBaseDeviceMemory::uploadSync(const void *src, size_t size, size_t offset)
{
    CudaDriver::throwOnCudaError(cuMemcpyHtoD((CUdeviceptr)m_ptr + offset, src, size), __FILE__, __LINE__);
}

void CudaBaseDeviceMemory::downloadSync(void *dst, size_t size, size_t offset) const
{
    CudaDriver::throwOnCudaError(cuMemcpyDtoH(dst, (CUdeviceptr)m_ptr + offset, size), __FILE__, __LINE__);
}


CudaConstantMemory::CudaConstantMemory(void *ptr, unsigned size)
{
    m_ptr = ptr;
    m_size = size;
    m_reserved = size;
}

CudaConstantMemory::~CudaConstantMemory()
{

}


void CudaConstantMemory::resize(size_t)
{
    throw std::runtime_error("Can not resize constant memory!");
}




CudaDeviceMemory::CudaDeviceMemory()
{
}

CudaDeviceMemory::~CudaDeviceMemory()
{
	if (m_ptr != NULL)
		cuMemFree((CUdeviceptr)m_ptr);
}

void CudaDeviceMemory::resize(size_t size)
{
    if (size > m_reserved) {
        if (m_ptr != NULL)
            cuMemFree((CUdeviceptr)m_ptr);
        m_reserved = size;
        CudaDriver::throwOnCudaError(cuMemAlloc((CUdeviceptr*)&m_ptr, m_reserved), __FILE__, __LINE__);
    }
    m_size = size;
}


}
