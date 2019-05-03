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

#include "CuDnnTensorDescriptor.h"

#include "CuDnnError.h"


#include <stdexcept>

namespace cudnn {

CuDnnTensorDescriptor::CuDnnTensorDescriptor()
{
    throwCudnnErrorImpl(cudnnCreateTensorDescriptor(&m_desc));
}

CuDnnTensorDescriptor::~CuDnnTensorDescriptor()
{
    checkCudnnError(cudnnDestroyTensorDescriptor(m_desc));
}

void CuDnnTensorDescriptor::setupTightlyPacked(const std::vector<unsigned> &dimensions, cudnnDataType_t dataType)
{
    if (dimensions.empty())
        throw std::runtime_error("Tensor must have at least one dimension!");

    std::vector<unsigned> strides;
    strides.resize(dimensions.size());
    strides.back() = 1;

    for (int i = strides.size()-2; i >= 0; i--) {
        strides[i] = strides[i+1] * dimensions[i+1];
    }

    setup(dimensions, strides, dataType);
}

void CuDnnTensorDescriptor::setup(const std::vector<unsigned> &dimensions, const std::vector<unsigned> &strides, cudnnDataType_t dataType)
{
    if (strides.size() != dimensions.size())
        throw std::runtime_error("dimensions and strides must match!");

    throwCudnnErrorImpl(cudnnSetTensorNdDescriptor(
        m_desc,
        dataType,
        dimensions.size(),
        (const int*) dimensions.data(),
        (const int*) strides.data()
    ));
}

}
