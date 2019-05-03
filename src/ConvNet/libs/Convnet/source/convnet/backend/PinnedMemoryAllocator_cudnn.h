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

#ifndef PINNEDMEMORYALLOCATOR_CUDNN_H
#define PINNEDMEMORYALLOCATOR_CUDNN_H

#include <map>
#include <mutex>
#include <vector>

namespace convnet {

class PinnedMemoryAllocator_cudnn
{
    public:
        ~PinnedMemoryAllocator_cudnn();

        void* allocate(std::size_t size, bool writeCombined = false);
        void free(void* ptr);
    protected:
        std::mutex m_mutex;
        struct Attribs {
            std::size_t size;
            bool writeCombined;
            inline bool operator<(const Attribs &rhs) const {
                if (size < rhs.size) return true;
                if (size > rhs.size) return false;
                if (writeCombined && !rhs.writeCombined) return true;
                if (!writeCombined && rhs.writeCombined) return false;
                return false;
            }
        };
        std::map<Attribs, std::vector<void*>> m_freeMemory;
        std::map<void*, Attribs> m_allocatedMemory;
};

}

#endif // PINNEDMEMORYALLOCATOR_CUDNN_H

