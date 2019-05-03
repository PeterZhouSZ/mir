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

#include "CuDnnDropoutDescriptor.h"

#include "CuDnnContext.h"
#include "CuDnnTensorDescriptor.h"

#include "CuDnnError.h"

#include <stdexcept>

namespace cudnn {

CuDnnDropoutDescriptor::CuDnnDropoutDescriptor()
{
    throwCudnnError(cudnnCreateDropoutDescriptor(&m_desc));
}

CuDnnDropoutDescriptor::~CuDnnDropoutDescriptor()
{
    checkCudnnError(cudnnDestroyDropoutDescriptor(m_desc));
}

std::size_t CuDnnDropoutDescriptor::getRngStateSize(const CuDnnContext &context) const
{
    std::size_t result;
    throwCudnnError(cudnnDropoutGetStatesSize(context.getHandle(), &result));
    return result;
}

std::size_t CuDnnDropoutDescriptor::getSwitchesSize(const CuDnnContext &/*context*/, const CuDnnTensorDescriptor &tensor) const
{
    std::size_t result;
    throwCudnnError(cudnnDropoutGetReserveSpaceSize(tensor.getDescriptor(), &result));
    return result;
}


void CuDnnDropoutDescriptor::setup(
            const CuDnnContext &context,
            float dropout,
            void *rngState,
            unsigned rngStateSize,
            unsigned rngSeed)
{
    throwCudnnError(cudnnSetDropoutDescriptor(
            m_desc,
            context.getHandle(),
            dropout,
            rngState,
            rngStateSize,
            rngSeed
    ));
}


}
