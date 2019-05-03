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

#include "PinnedMemoryAllocator_cudnn.h"

#include <iostream>
#include <stdexcept>

#include <cuda.h>
#include <cudaUtils/CudaDriver.h>

namespace convnet {

PinnedMemoryAllocator_cudnn::~PinnedMemoryAllocator_cudnn()
{
    if (!m_allocatedMemory.empty())
        std::cerr << m_allocatedMemory.size() << " pinned memory allocations not freed!" << std::endl;

    for (auto &pair : m_freeMemory) {
        for (auto p : pair.second)
            cuMemFreeHost(p);
    }
}

void* PinnedMemoryAllocator_cudnn::allocate(std::size_t size, bool writeCombined)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    void *ptr;

    auto it = m_freeMemory.find({size, writeCombined});
    if ((it != m_freeMemory.end()) && (!it->second.empty())) {
        ptr = it->second.back();
        it->second.pop_back();
    } else {
        std::cout << "Allocating " << size << " bytes of " << (writeCombined?"write combined":"cached") << " memory!" << std::endl;
        std::cout << m_freeMemory.size() << " free allocations so far" << std::endl;
        for (const auto &pair : m_freeMemory) {
            std::cout << pair.second.size() << "x   " << pair.first.size << " bytes " << (pair.first.writeCombined?"write combined":"cached") << std::endl;
        }
        CudaUtils::CudaDriver::throwOnCudaError(cuMemHostAlloc(&ptr, size, writeCombined?CU_MEMHOSTALLOC_WRITECOMBINED:0), __FILE__, __LINE__);
    }
    m_allocatedMemory[ptr] = {size, writeCombined};
    return ptr;
}

void PinnedMemoryAllocator_cudnn::free(void* ptr)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_allocatedMemory.find(ptr);
    if (it != m_allocatedMemory.end()) {
        m_freeMemory[it->second].push_back(it->first);
        m_allocatedMemory.erase(it);
    } else
        throw std::runtime_error("Double free or pointer corruption!");

}

}
