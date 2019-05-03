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

#include "CuDnnConvolutionDescriptor.h"

#include "CuDnnError.h"

#include <stdexcept>

namespace cudnn {

CuDnnConvolutionDescriptor::CuDnnConvolutionDescriptor()
{
    throwCudnnError(cudnnCreateConvolutionDescriptor(&m_desc));
}

CuDnnConvolutionDescriptor::~CuDnnConvolutionDescriptor()
{
    checkCudnnError(cudnnDestroyConvolutionDescriptor(m_desc));
}


void CuDnnConvolutionDescriptor::setup(const std::vector<unsigned> &padding, const std::vector<unsigned> &stride, const std::vector<unsigned> &upscale,
                    cudnnDataType_t dataType, cudnnConvolutionMode_t convolutionMode)
{
    if (padding.size() != stride.size())
        throw std::runtime_error("Sizes (dimensions) of padding, stride, and upscale must match!");

    if (padding.size() != upscale.size())
        throw std::runtime_error("Sizes (dimensions) of padding, stride, and upscale must match!");

    throwCudnnError(cudnnSetConvolutionNdDescriptor(
            m_desc,
            padding.size(),
            (const int*)padding.data(),
            (const int*)stride.data(),
            (const int*)upscale.data(),
            convolutionMode,
            dataType
    ));

}


}
