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

#include "CudaDeviceContext.h"

#include "CudaDriver.h"
#include "CudaDevice.h"

namespace CudaUtils {


CudaDeviceContext::CudaDeviceContext(CudaDevice *device) : m_device(device)
{
    CudaDriver::throwOnCudaError(cuCtxCreate(&m_contextHandle, CU_CTX_SCHED_AUTO, m_device->getDeviceHandle()), __FILE__, __LINE__);
    CudaDriver::throwOnCudaError(cuCtxPopCurrent(NULL), __FILE__, __LINE__);
}

CudaDeviceContext::~CudaDeviceContext()
{
    CudaDriver::throwOnCudaError(cuCtxDestroy(m_contextHandle), __FILE__, __LINE__); // note that this sigterms the process on failure
}


void CudaDeviceContext::makeCurrent()
{
    CudaDriver::throwOnCudaError(cuCtxSetCurrent(m_contextHandle), __FILE__, __LINE__);
}

void CudaDeviceContext::setBankSize4Byte()
{
    CudaDriver::throwOnCudaError(cuCtxPushCurrent(m_contextHandle), __FILE__, __LINE__);
    CudaDriver::throwOnCudaError(cuCtxSetSharedMemConfig(CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE), __FILE__, __LINE__);
    CudaDriver::throwOnCudaError(cuCtxPopCurrent(NULL), __FILE__, __LINE__);
}

void CudaDeviceContext::setPreferredSharedMem()
{
    CudaDriver::throwOnCudaError(cuCtxPushCurrent(m_contextHandle), __FILE__, __LINE__);
    CudaDriver::throwOnCudaError(cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_SHARED), __FILE__, __LINE__);
    CudaDriver::throwOnCudaError(cuCtxPopCurrent(NULL), __FILE__, __LINE__);
}

void CudaDeviceContext::setPreferredL1Cache()
{
    CudaDriver::throwOnCudaError(cuCtxPushCurrent(m_contextHandle), __FILE__, __LINE__);
    CudaDriver::throwOnCudaError(cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1), __FILE__, __LINE__);
    CudaDriver::throwOnCudaError(cuCtxPopCurrent(NULL), __FILE__, __LINE__);
}


}
