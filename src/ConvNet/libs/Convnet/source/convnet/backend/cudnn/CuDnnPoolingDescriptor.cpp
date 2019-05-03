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

#include "CuDnnPoolingDescriptor.h"

#include "CuDnnError.h"

#include <stdexcept>

namespace cudnn {

CuDnnPoolingDescriptor::CuDnnPoolingDescriptor()
{
    throwCudnnError(cudnnCreatePoolingDescriptor(&m_desc));
}

CuDnnPoolingDescriptor::~CuDnnPoolingDescriptor()
{
    checkCudnnError(cudnnDestroyPoolingDescriptor(m_desc));
}

void CuDnnPoolingDescriptor::setup(const std::vector<unsigned> &window, const std::vector<unsigned> &padding, const std::vector<unsigned> &stride,
            cudnnPoolingMode_t mode, cudnnNanPropagation_t maxPoolNanProp)
{
    if (padding.size() != stride.size())
        throw std::runtime_error("Sizes (dimensions) of padding, stride, and window size must match!");

    if (padding.size() != window.size())
        throw std::runtime_error("Sizes (dimensions) of padding, stride, and window size must match!");


    throwCudnnError(cudnnSetPoolingNdDescriptor(
        m_desc,
        mode,
        maxPoolNanProp,
        window.size(),
        (const int*)window.data(),
        (const int*)padding.data(),
        (const int*)stride.data()
    ));
}



}
