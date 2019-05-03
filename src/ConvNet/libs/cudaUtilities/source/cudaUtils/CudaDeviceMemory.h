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

#ifndef CUDA_DEVICE_MEMORY_H_
#define CUDA_DEVICE_MEMORY_H_

#include "CudaBaseMemoryChunk.h"
#include "CudaStream.h"

namespace CudaUtils {


/**
 * @brief Encapsulates linear video memory
 */
class CudaBaseDeviceMemory : public CudaBaseMemoryChunk
{
    public:
        /// Asynchronous upload from linear system memory into the linear video memory
        void uploadAsync(const void *src, size_t size, size_t offset, const CudaStream &stream);
        /// Asynchronous download from linear video memory into the linear system memory
        void downloadAsync(void *dst, size_t size, size_t offset, const CudaStream &stream) const;

        void uploadSync(const void *src, size_t size, size_t offset);
        void downloadSync(void *dst, size_t size, size_t offset) const;
};


class CudaConstantMemory : public CudaBaseDeviceMemory
{
    public:
        CudaConstantMemory(void *ptr, unsigned size);
        virtual ~CudaConstantMemory();

        virtual void resize(size_t size) override;
};

/**
 * @brief Encapsulates linear video memory
 */
class CudaDeviceMemory : public CudaBaseDeviceMemory
{
    public:
        CudaDeviceMemory();
        virtual ~CudaDeviceMemory();

        /// @copydoc BaseMemoryChunk::resize
        virtual void resize(size_t size) override;
};

}

#endif // CUDA_DEVICE_MEMORY_H_
