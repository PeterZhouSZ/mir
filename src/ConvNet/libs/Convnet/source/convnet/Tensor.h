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

#ifndef TENSOR_H
#define TENSOR_H

#include <cstdint>
#include <ostream>

namespace convnet {

class ExecutionStream;
class ConvNet;

enum DataFormat {
    DF_UInt8,
    DF_FLOAT16,
    DF_FLOAT32
};

struct TensorSize {
    unsigned width = 0;
    unsigned height = 0;
    unsigned depth = 0;
    unsigned channels = 0;

    TensorSize() = default;

    TensorSize(unsigned w, unsigned h, unsigned d, unsigned c) {
        width = w;
        height = h;
        depth = d;
        channels = c;
    }

    inline bool operator==(const TensorSize &rhs) const {
        return (width == rhs.width) && (height == rhs.height) && (depth == rhs.depth) && (channels == rhs.channels);
    }

    inline bool operator!=(const TensorSize &rhs) const {
        return !operator==(rhs);
    }

    inline unsigned numElements() const {
        return width * height * depth * channels;
    }
};


inline std::ostream &operator<<(std::ostream &stream, const TensorSize &dim) {
    stream
        << "width: " << dim.width
        << " height: " << dim.height
        << " depth: " << dim.depth
        << " channels: " << dim.channels;
    return stream;
}


class MappedTensor
{
    public:
        MappedTensor() { }
        MappedTensor(void *ptr, const TensorSize &size, unsigned numInstances, DataFormat format);

        inline const TensorSize &getSize() const { return m_size; }
        inline unsigned getNumInstances() const { return m_numInstances; }
        inline DataFormat getFormat() const { return m_format; }

        inline std::size_t getIndex(unsigned x, unsigned y, unsigned z, unsigned c, unsigned n) const {
            return x + y * m_strides[0] + z * m_strides[1] + c * m_strides[2] + n * m_strides[3];
        }

        template<typename Type>
        inline Type &get(unsigned x, unsigned y, unsigned z, unsigned c, unsigned n) {
            return reinterpret_cast<Type*>(m_ptr)[getIndex(x, y, z, c, n)];
        }
        template<typename Type>
        inline const Type &get(unsigned x, unsigned y, unsigned z, unsigned c, unsigned n) const {
            return reinterpret_cast<Type*>(m_ptr)[getIndex(x, y, z, c, n)];
        }

        template<typename Type>
        inline Type* data(unsigned instance) { return reinterpret_cast<Type*>(m_ptr) + instance * m_strides[3]; }

        template<typename Type>
        inline const Type* data(unsigned instance) const { return reinterpret_cast<const Type*>(m_ptr) + instance * m_strides[3]; }
    protected:
        TensorSize m_size;
        std::size_t __restrict__ m_strides[4];
        unsigned m_numInstances = 0;
        DataFormat m_format;
        void * __restrict__ m_ptr = nullptr;
};


class Tensor
{
    public:
        Tensor(ConvNet &convnet) : m_convnet(convnet) { }
        virtual ~Tensor() = default;

        virtual void allocate(const TensorSize &size, unsigned numInstances, DataFormat format) = 0;

        virtual MappedTensor lock(bool discard = false) = 0;
        virtual void unlock(MappedTensor mappedTensor, bool dirty = true) = 0;

        virtual void setZero(ExecutionStream &stream) = 0;
        virtual void setZeroSync() = 0;

        inline const TensorSize &getSize() const { return m_size; }
        inline unsigned getNumInstances() const { return m_numInstances; }
        inline DataFormat getFormat() const { return m_format; }
        virtual std::size_t getMemoryRequirements() const;
        inline unsigned getTotalNumberOfElements() const { return m_size.numElements() * m_numInstances; }
    protected:
        ConvNet &m_convnet;

        unsigned m_numInstances;
        TensorSize m_size;
        DataFormat m_format;
};

}

#endif // TENSOR_H
