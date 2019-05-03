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

#include "Tensor.h"


#include <stdexcept>

namespace convnet {

MappedTensor::MappedTensor(void *ptr, const TensorSize &size, unsigned numInstances, DataFormat format) : m_size(size), m_numInstances(numInstances), m_format(format), m_ptr(ptr)
{
    m_strides[0] = size.width;
    m_strides[1] = size.height * m_strides[0];
    m_strides[2] = size.depth * m_strides[1];
    m_strides[3] = size.channels * m_strides[2];
}


std::size_t Tensor::getMemoryRequirements() const
{
    switch (m_format) {
        case DF_UInt8:
            return m_size.numElements() * m_numInstances * 1;
        case DF_FLOAT16:
            return m_size.numElements() * m_numInstances * 2;
        case DF_FLOAT32:
            return m_size.numElements() * m_numInstances * 4;
        default:
            throw std::runtime_error("Unhandled data format!");
    }
}

}
