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

#ifndef BASEMEMORYCHUNK_H
#define BASEMEMORYCHUNK_H

#include <stdint.h>
#include <stddef.h>

namespace CudaUtils {

/**
 * @brief Base class for pinned system and linear video memory
 * @details
 */
class CudaBaseMemoryChunk
{
    public:
        /// Ctor
        CudaBaseMemoryChunk();
        /// Dtor, frees the allocated memory
        virtual ~CudaBaseMemoryChunk();
        /// Prevent copy constructing
        CudaBaseMemoryChunk(const CudaBaseMemoryChunk &) = delete;
        /// Prevent copying
        const CudaBaseMemoryChunk &operator=(const CudaBaseMemoryChunk &) = delete;

        
        /// Makes sure that at least size bytes of memory are allocated
        virtual void resize(size_t size) = 0;

        /// Returns the size in bytes last set via @ref resize resize
        inline size_t size() const { return m_size; }
        /// Returns the currently allocated size in bytes
        inline size_t reserved() const { return m_reserved; }

        /// Const pointer to the memory
        inline const void *getPtr() const { return m_ptr; }
        /// Non-const pointer to the memory
        inline void *getPtr() { return m_ptr; }
    protected:
        /// Pointer to the allocated memory. Can be a device pointer.
        void *m_ptr;
        /// Amount of memory in bytes requested in the last @ref resize call
        size_t m_size;
        /// Amount of memory actually allocated in bytes
        size_t m_reserved;

};
}

#endif // BASEMEMORYCHUNK_H
