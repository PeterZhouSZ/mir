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

#include "CudaFence.h"
#include "CudaDriver.h"

#include <cuda.h>

namespace CudaUtils {


CudaFence::CudaFence(unsigned flags)
{
    CudaDriver::throwOnCudaError(cuEventCreate(&m_event, flags), __FILE__, __LINE__);
}

CudaFence::~CudaFence()
{
    CudaDriver::throwOnCudaError(cuEventDestroy(m_event), __FILE__, __LINE__);
}


void CudaFence::insertIntoStream(const CudaStream &stream)
{
    CudaDriver::throwOnCudaError(cuEventRecord(m_event, stream.getHandle()), __FILE__, __LINE__);
}

bool CudaFence::query()
{
    CUresult state = cuEventQuery(m_event);
    switch (state) {
        case CUDA_SUCCESS:
            return true;
        case CUDA_ERROR_NOT_READY:
            return false;
        default:
            CudaDriver::throwOnCudaError(state, __FILE__, __LINE__);
            return false;
    }
}

void CudaFence::waitFor()
{
    CudaDriver::throwOnCudaError(cuEventSynchronize(m_event), __FILE__, __LINE__);
}

void CudaFence::makeStreamWaitOnFence(const CudaStream &stream)
{
    CudaDriver::throwOnCudaError(cuStreamWaitEvent(stream.getHandle(), m_event, 0), __FILE__, __LINE__);
}


}
