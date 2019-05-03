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

#ifndef CUDNNWRAPPER_H
#define CUDNNWRAPPER_H

#include "cudnn/CuDnnContext.h"
#include "CudnnAuxKernels.h"
#include "PinnedMemoryAllocator_cudnn.h"

#include <cudaUtils/CudaStream.h>


namespace convnet {

class CuDnnWrapper
{
    public:
        CuDnnWrapper();

        inline const CudnnAuxKernels &getAuxKernels() const { return m_auxKernels; }
        inline PinnedMemoryAllocator_cudnn &getPinnedMemoryAllocator() { return m_pinnedMemoryAllocator; }
    protected:
        CudnnAuxKernels m_auxKernels;
        PinnedMemoryAllocator_cudnn m_pinnedMemoryAllocator;
};

}

#endif // CUDNNWRAPPER_H
