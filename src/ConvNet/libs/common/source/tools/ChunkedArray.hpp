/*
 * Common Utilities - Distributed for "Mental Image Retrieval" implementation
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

#ifndef _CHUNKED_ARRAY_HPP_
#define _CHUNKED_ARRAY_HPP_

#include <vector>
#include <stdint.h>
#include <iterator>

#include <stdexcept>
#include <memory>

/** @addtogroup Codebase_Group
 *  @{
 */

namespace ChunkedArrayDetail {
template<typename MaybeConstType, class MaybeConstArray>
class Iterator : public std::iterator< std::forward_iterator_tag, MaybeConstType >
{
    public:
        Iterator() {
            makeInvalid();
        }

        Iterator(const Iterator<MaybeConstType, MaybeConstArray> &other) = default;
        Iterator<MaybeConstType, MaybeConstArray> &operator=(const Iterator<MaybeConstType, MaybeConstArray> &other) = default;

        bool operator==(const Iterator<MaybeConstType, MaybeConstArray> &other) const {
            return (m_array == other.m_array) &&
                   (m_chunk == other.m_chunk) &&
                   (m_slot == other.m_slot);
        }
        bool operator!=(const Iterator<MaybeConstType, MaybeConstArray> &other) const {
            return !(*this == other);
        }

        void makeInvalid() {
            m_chunk = -1;
            m_slot = -1;
            m_array = nullptr;
        }

        inline MaybeConstType &operator*() {
            return *(m_array->m_chunks[m_chunk]->slots()+m_slot);
        }
        inline MaybeConstType *operator->() {
            return m_array->m_chunks[m_chunk]->slots()+m_slot;
        }

        inline void increment() {
            do {
                if (++m_slot >= MaybeConstArray::CHUNK_SIZE) {
                    m_slot = 0;
                    do {
                        ++m_chunk;
                        if (m_chunk >= m_array->m_slotMasks.size()) {
                            makeInvalid();
                            return;
                        }
                    } while (m_array->m_slotMasks[m_chunk] == 0);
                }
            } while ((m_array->m_slotMasks[m_chunk] & (1ul << (uint64_t)m_slot)) == 0);
        }

        Iterator<MaybeConstType, MaybeConstArray> operator++() {
            increment();
            return *this;
        }

        Iterator<MaybeConstType, MaybeConstArray> operator++(int) {
            Iterator<MaybeConstType, MaybeConstArray> other(*this);
            increment();
            return other;
        }


        inline void setToFirstOf(MaybeConstArray *array) {
            m_array = array;
            if (m_array->m_slotMasks.empty()) {
                makeInvalid();
                return;
            }
            m_chunk = 0;
            m_slot = 0;
            if ((m_array->m_slotMasks[0] & 1ul) == 0ul)
                increment();
        }
    private:
        MaybeConstArray *m_array;
        unsigned m_chunk;
        unsigned m_slot;
};

}


template<unsigned x>
constexpr unsigned NoMoreThan64() {
    static_assert(x<=64u, "ChunkSize must be <= 64");
    return x;
}

template<typename Type, unsigned ChunkSize=64>
class ChunkedArray
{
    public:
        enum {
            CHUNK_SIZE = NoMoreThan64<ChunkSize>()
        };

        template<typename... Args>
        unsigned allocate(Args&&... args);

        int findIndexOf(const Type *elem) const;
        void free(unsigned index);
        void free(Type *elem);


        inline Type &operator[](unsigned index) {
            const unsigned chunkIndex = index / CHUNK_SIZE;
            const unsigned slotIndex = index % CHUNK_SIZE;
            return m_chunks[chunkIndex]->slots()[slotIndex];
        }

        inline const Type &operator[](unsigned index) const {
            const unsigned chunkIndex = index / CHUNK_SIZE;
            const unsigned slotIndex = index % CHUNK_SIZE;
            return m_chunks[chunkIndex]->slots()[slotIndex];
        }

        inline bool inUse(unsigned index) const {
            const unsigned chunkIndex = index / CHUNK_SIZE;
            const unsigned slotIndex = index % CHUNK_SIZE;
            return m_slotMasks[chunkIndex] & (1ul << (uint64_t)slotIndex);
        }

        ChunkedArray(const ChunkedArray&) = delete;
        ChunkedArray &operator=(const ChunkedArray&) = delete;

        ~ChunkedArray();
        void clear();

        inline unsigned reservedSize() const { return m_chunks.size() * CHUNK_SIZE; }

        typedef ChunkedArrayDetail::Iterator<Type, ChunkedArray<Type, ChunkSize> > iterator;
        typedef ChunkedArrayDetail::Iterator<const Type, const ChunkedArray<Type, ChunkSize> > const_iterator;

        inline iterator begin() {
            iterator iter;
            iter.setToFirstOf(this);
            return iter;
        }

        inline iterator end() {
            iterator iter;
            return iter;
        }

        inline const_iterator begin() const {
            const_iterator iter;
            iter.setToFirstOf(this);
            return iter;
        }

        inline const_iterator end() const {
            const_iterator iter;
            return iter;
        }

    private:
        struct Chunk {
            alignas(Type) unsigned char data[CHUNK_SIZE * sizeof(Type)];

            inline Type *slots() { return ((Type*)data); }
            inline const Type *slots() const { return ((const Type*)data); }
        };
        std::vector<std::unique_ptr<Chunk>> m_chunks;
        std::vector<uint64_t> m_slotMasks;

        friend iterator;
        friend const_iterator;
};


template<typename Type, unsigned ChunkSize>
ChunkedArray<Type, ChunkSize>::~ChunkedArray()
{
    clear();
}


template<typename Type, unsigned ChunkSize>
void ChunkedArray<Type, ChunkSize>::clear()
{
    for (unsigned i = 0; i < m_slotMasks.size(); i++) {
        if (m_slotMasks[i] == 0)
            continue;

        for (unsigned j = 0; j < CHUNK_SIZE; j++)
            if ((m_slotMasks[i] & (1l << (uint64_t)j)) != 0) {
                m_chunks[i]->slots()[j].~Type();
            }

        m_slotMasks[i] = 0;
    }
}


template<typename Type, unsigned ChunkSize>
template<typename... Args>
unsigned ChunkedArray<Type, ChunkSize>::allocate(Args&&... args)
{
    for (unsigned i = 0; i < m_slotMasks.size(); i++) {
        if (m_slotMasks[i] == (uint64_t)-1l)
            continue;

        for (unsigned j = 0; j < CHUNK_SIZE; j++)
            if ((m_slotMasks[i] & (1l << (uint64_t)j)) == 0) {
                m_slotMasks[i] |= (1l << (uint64_t)j);
                new (&m_chunks[i]->slots()[j]) Type(std::forward<Args>(args)...);
                return i*CHUNK_SIZE+j;
            }
    }
    m_chunks.push_back(new Chunk());
    m_slotMasks.push_back(1);
    new (&m_chunks[m_chunks.size()-1]->slots()[0]) Type(std::forward<Args>(args)...);
    return (m_chunks.size()-1)*CHUNK_SIZE+0;
}


template<typename Type, unsigned ChunkSize>
void ChunkedArray<Type, ChunkSize>::free(unsigned index)
{
    const unsigned chunkIndex = index / CHUNK_SIZE;
    const unsigned slotIndex = index % CHUNK_SIZE;

    m_slotMasks[chunkIndex] &= ~(1l << (uint64_t)slotIndex);
    m_chunks[chunkIndex]->slots()[slotIndex].~Type();
}

template<typename Type, unsigned ChunkSize>
void ChunkedArray<Type, ChunkSize>::free(Type *elem)
{
    for (unsigned i = 0; i < m_chunks.size(); i++) {
        if ((elem >= m_chunks[i]->slots()) && (elem < (m_chunks[i]->slots() + CHUNK_SIZE))) {
            elem->~Type();
            unsigned index = std::size_t(elem - m_chunks[i]->slots());
            m_slotMasks[i] &= ~(1l << (uint64_t)index);
        }
    }
}

template<typename Type, unsigned ChunkSize>
int ChunkedArray<Type, ChunkSize>::findIndexOf(const Type *elem) const
{
    for (unsigned i = 0; i < m_chunks.size(); i++) {
        if ((elem >= m_chunks[i]->slots()) && (elem < (m_chunks[i]->slots() + CHUNK_SIZE))) {
            unsigned index = std::size_t(elem - m_chunks[i]->slots());
            return i*CHUNK_SIZE + index;
        }
    }
    return -1;
}

/// @}

#endif // _CHUNKED_ARRAY_HPP_

