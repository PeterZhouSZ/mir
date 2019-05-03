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

#ifndef EXECUTIONWORKSPACEIMPL_CUDNN_H
#define EXECUTIONWORKSPACEIMPL_CUDNN_H

#include "../ExecutionWorkspace.h"
#include "cudnn/CuDnnContext.h"
#include "CudnnAuxKernels.h"


#include <cudaUtils/CudaStream.h>
#include <cudaUtils/CudaDeviceMemory.h>

#include <cstdint>

namespace convnet {

class ExecutionWorkspaceImpl_cudnn : public ExecutionWorkspace
{
    public:
        ExecutionWorkspaceImpl_cudnn(std::size_t workspaceSize);

        inline CudaUtils::CudaDeviceMemory &getWorkspaceMemory() { return m_workspaceMemory; }
    protected:
        CudaUtils::CudaDeviceMemory m_workspaceMemory;
};

}

#endif // EXECUTIONWORKSPACEIMPL_CUDNN_H
