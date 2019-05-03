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

#include "CudaCodeModule.h"
#include "CudaDriver.h"
#include "CudaKernel.h"
/*
#include "CudaTextureReference.h"
#include "CudaSurfaceReference.h"
*/
#include "CudaDeviceMemory.h"

namespace CudaUtils {


CudaCodeModule::CudaCodeModule()
{
    m_handleValid = false;
}

CudaCodeModule::~CudaCodeModule()
{
    if (m_handleValid)
        CudaDriver::throwOnCudaError(cuModuleUnload(m_moduleHandle), __FILE__, __LINE__);
}

void CudaCodeModule::loadFromFile(const char *filename)
{
    if (m_handleValid)
        CudaDriver::throwOnCudaError(cuModuleUnload(m_moduleHandle), __FILE__, __LINE__);
    m_handleValid = false;

    CudaDriver::throwOnCudaError(cuModuleLoad(&m_moduleHandle, filename), __FILE__, __LINE__);
    m_handleValid = true;
}

void CudaCodeModule::loadFromMemory(const void *ptr)
{
    if (m_handleValid)
        CudaDriver::throwOnCudaError(cuModuleUnload(m_moduleHandle), __FILE__, __LINE__);
    m_handleValid = false;

    CudaDriver::throwOnCudaError(cuModuleLoadData(&m_moduleHandle, ptr), __FILE__, __LINE__);
    m_handleValid = true;
}

CudaKernel *CudaCodeModule::getKernel(const char *kernelName)
{
    CUfunction functionHandle;
    CudaDriver::throwOnCudaError(cuModuleGetFunction(&functionHandle, m_moduleHandle, kernelName), __FILE__, __LINE__);

    return new CudaKernel(functionHandle);
}

/*
CudaTextureReference *CudaCodeModule::getTexReference(const char *name)
{
    CUtexref handle;
    CudaDriver::throwOnCudaError(cuModuleGetTexRef(&handle, m_moduleHandle, name), __FILE__, __LINE__);

    return new CudaTextureReference(handle);
}


CudaSurfaceReference *CudaCodeModule::getSurfReference(const char *name)
{
    CUsurfref handle;
    CudaDriver::throwOnCudaError(cuModuleGetSurfRef(&handle, m_moduleHandle, name), __FILE__, __LINE__);

    return new CudaSurfaceReference(handle);
}

CudaConstantMemory *CudaCodeModule::getConstantMemory(const char *name)
{
    CUdeviceptr ptr;
    size_t bytes;
    CudaDriver::throwOnCudaError(cuModuleGetGlobal(&ptr, &bytes, m_moduleHandle, name), __FILE__, __LINE__);

    return new CudaConstantMemory((void*)ptr, bytes);
}
*/

}
