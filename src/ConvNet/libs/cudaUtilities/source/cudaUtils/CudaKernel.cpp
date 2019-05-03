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

#include "CudaKernel.h"
#include "CudaDriver.h"
#include "CudaStream.h"

#include <assert.h>

namespace CudaUtils {

CudaKernel::CudaKernel(CUfunction functionHandle) : m_functionHandle(functionHandle)
{
    //ctor
}

CudaKernel::~CudaKernel()
{
    //dtor
}

void CudaKernel::launch(const std::array<unsigned, 3> &blockDim, const std::array<unsigned, 3> &gridDim, void *params, size_t paramSize, const CudaStream &stream, unsigned dynamicSharedMemSize) const
{
    void *config[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER, params,
        CU_LAUNCH_PARAM_BUFFER_SIZE, &paramSize,
        CU_LAUNCH_PARAM_END
    };
    CUresult status = cuLaunchKernel(m_functionHandle,
                            gridDim[0], gridDim[1], gridDim[2],
                            blockDim[0], blockDim[1], blockDim[2],
                            dynamicSharedMemSize, stream.getHandle(), NULL, config);

    CudaDriver::throwOnCudaError(status, __FILE__, __LINE__);
}


}
