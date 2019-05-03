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

#include "CudaDriver.h"
#include <stdexcept>
#include <sstream>

#include <cuda.h>

#include "CudaDevice.h"


namespace CudaUtils {

const char* CudaDriver::cvtCUresult2string(unsigned result)
{
    switch (result) {
        case CUDA_SUCCESS: return "CUDA_SUCCESS";
        case CUDA_ERROR_INVALID_VALUE: return "CUDA_ERROR_INVALID_VALUE";
        case CUDA_ERROR_OUT_OF_MEMORY: return "CUDA_ERROR_OUT_OF_MEMORY";
        case CUDA_ERROR_NOT_INITIALIZED: return "CUDA_ERROR_NOT_INITIALIZED";
        case CUDA_ERROR_DEINITIALIZED: return "CUDA_ERROR_DEINITIALIZED";
        case CUDA_ERROR_PROFILER_DISABLED: return "CUDA_ERROR_PROFILER_DISABLED";
        case CUDA_ERROR_PROFILER_NOT_INITIALIZED: return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
        case CUDA_ERROR_PROFILER_ALREADY_STARTED: return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
        case CUDA_ERROR_PROFILER_ALREADY_STOPPED: return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
        case CUDA_ERROR_NO_DEVICE: return "CUDA_ERROR_NO_DEVICE";
        case CUDA_ERROR_INVALID_DEVICE: return "CUDA_ERROR_INVALID_DEVICE";
        case CUDA_ERROR_INVALID_IMAGE: return "CUDA_ERROR_INVALID_IMAGE";
        case CUDA_ERROR_INVALID_CONTEXT: return "CUDA_ERROR_INVALID_CONTEXT";
        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
        case CUDA_ERROR_MAP_FAILED: return "CUDA_ERROR_MAP_FAILED";
        case CUDA_ERROR_UNMAP_FAILED: return "CUDA_ERROR_UNMAP_FAILED";
        case CUDA_ERROR_ARRAY_IS_MAPPED: return "CUDA_ERROR_ARRAY_IS_MAPPED";
        case CUDA_ERROR_ALREADY_MAPPED: return "CUDA_ERROR_ALREADY_MAPPED";
        case CUDA_ERROR_NO_BINARY_FOR_GPU: return "CUDA_ERROR_NO_BINARY_FOR_GPU";
        case CUDA_ERROR_ALREADY_ACQUIRED: return "CUDA_ERROR_ALREADY_ACQUIRED";
        case CUDA_ERROR_NOT_MAPPED: return "CUDA_ERROR_NOT_MAPPED";
        case CUDA_ERROR_NOT_MAPPED_AS_ARRAY: return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
        case CUDA_ERROR_NOT_MAPPED_AS_POINTER: return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
        case CUDA_ERROR_ECC_UNCORRECTABLE: return "CUDA_ERROR_ECC_UNCORRECTABLE";
        case CUDA_ERROR_UNSUPPORTED_LIMIT: return "CUDA_ERROR_UNSUPPORTED_LIMIT";
        case CUDA_ERROR_CONTEXT_ALREADY_IN_USE: return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
        case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED: return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
        case CUDA_ERROR_INVALID_SOURCE: return "CUDA_ERROR_INVALID_SOURCE";
        case CUDA_ERROR_FILE_NOT_FOUND: return "CUDA_ERROR_FILE_NOT_FOUND";
        case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
        case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
        case CUDA_ERROR_OPERATING_SYSTEM: return "CUDA_ERROR_OPERATING_SYSTEM";
        case CUDA_ERROR_INVALID_HANDLE: return "CUDA_ERROR_INVALID_HANDLE";
        case CUDA_ERROR_NOT_FOUND: return "CUDA_ERROR_NOT_FOUND";
        case CUDA_ERROR_NOT_READY: return "CUDA_ERROR_NOT_READY";
        case CUDA_ERROR_LAUNCH_FAILED: return "CUDA_ERROR_LAUNCH_FAILED";
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
        case CUDA_ERROR_LAUNCH_TIMEOUT: return "CUDA_ERROR_LAUNCH_TIMEOUT";
        case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
        case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
        case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
        case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
        case CUDA_ERROR_CONTEXT_IS_DESTROYED: return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
        case CUDA_ERROR_ASSERT: return "CUDA_ERROR_ASSERT";
        case CUDA_ERROR_TOO_MANY_PEERS: return "CUDA_ERROR_TOO_MANY_PEERS";
        case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED: return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
        case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED: return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
        case CUDA_ERROR_NOT_PERMITTED: return "CUDA_ERROR_NOT_PERMITTED";
        case CUDA_ERROR_NOT_SUPPORTED: return "CUDA_ERROR_NOT_SUPPORTED";
        case CUDA_ERROR_UNKNOWN: return "CUDA_ERROR_UNKNOWN";
        default: return "Undefined error code!";
    }
}


void CudaDriver::throwOnCudaError(unsigned result, const char *filename, unsigned lineNumber)
{
    if (result != CUDA_SUCCESS) {
        std::stringstream str;
        str << "Error in " << filename << ": " << lineNumber << ": Call returned " << cvtCUresult2string(result);
        throw std::runtime_error(str.str());
    }
}



uint32_t CudaDriver::m_singleInstanceCheck = 0;

CudaDriver::CudaDriver()
{
    try {
        if (__sync_fetch_and_add(&m_singleInstanceCheck, 1) != 0)
            throw std::runtime_error("The CudaDriver class was already instantiated!");


        throwOnCudaError(cuInit(0), __FILE__, __LINE__);

    } catch (...) {
        __sync_fetch_and_sub(&m_singleInstanceCheck, 1);
        throw;
    }
}

CudaDriver::~CudaDriver()
{
    __sync_fetch_and_sub(&m_singleInstanceCheck, 1);
}


unsigned CudaDriver::getNumDevices()
{
    int numDevices;
    throwOnCudaError(cuDeviceGetCount(&numDevices), __FILE__, __LINE__);

    return numDevices;
}


CudaDevice *CudaDriver::getMaxCCDevice()
{
    unsigned numDevices = getNumDevices();
    if (numDevices == 0)
        return NULL;

    CUdevice bestDeviceHandle;
    throwOnCudaError(cuDeviceGet(&bestDeviceHandle, 0), __FILE__, __LINE__);

    int bestMajor;
    int bestMinor;
    throwOnCudaError(cuDeviceGetAttribute(&bestMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, bestDeviceHandle), __FILE__, __LINE__);
    throwOnCudaError(cuDeviceGetAttribute(&bestMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, bestDeviceHandle), __FILE__, __LINE__);

    for (unsigned i = 1; i < numDevices; i++) {
        CUdevice deviceHandle;
        throwOnCudaError(cuDeviceGet(&deviceHandle, i), __FILE__, __LINE__);

        int major;
        int minor;
        throwOnCudaError(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, deviceHandle), __FILE__, __LINE__);
        throwOnCudaError(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, deviceHandle), __FILE__, __LINE__);

        if ((major > bestMajor) || ((major == bestMajor) && (minor > bestMinor))) {
            bestMinor = minor;
            bestMajor = major;
            bestDeviceHandle = deviceHandle;
        }
    }

    return new CudaDevice(this, bestDeviceHandle);
}


}
