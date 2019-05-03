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
#include "CuDnnActivationDescriptor.h"

#include "CuDnnError.h"

namespace cudnn {

CuDnnActivationDescriptor::CuDnnActivationDescriptor()
{
    throwCudnnError(cudnnCreateActivationDescriptor(&m_desc));
}

CuDnnActivationDescriptor::~CuDnnActivationDescriptor()
{
    checkCudnnError(cudnnDestroyActivationDescriptor(m_desc));
}

void CuDnnActivationDescriptor::setup(cudnnActivationMode_t mode, cudnnNanPropagation_t reluNanOpt, double reluCeiling)
{
    throwCudnnError(cudnnSetActivationDescriptor(m_desc, mode, reluNanOpt, reluCeiling));
}

}
