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

#include "CuDnnError.h"

#include <iostream>
#include <sstream>

namespace cudnn {

CuDnnError::CuDnnError(cudnnStatus_t error, const char *invocation, const char *file, int line)
{
    m_error = error;
    m_what = composeErrorMsg(error, invocation, file, line);
}


std::string CuDnnError::composeErrorMsg(cudnnStatus_t error, const char *invocation, const char *file, int line)
{
    std::stringstream msg;

    msg << "Cuda error " << error << ": " << cudnnGetErrorString(error);
    if (invocation != nullptr)
        msg << " calling " << invocation;
    if (file != nullptr)
        msg << " in " << file;
    if (line >= 0)
        msg << ":" << line;

    return msg.str();
}


const char* CuDnnError::what() const noexcept
{
    return m_what.c_str();
}


void checkCudnnErrorImpl(cudnnStatus_t error, const char *invocation, const char *file, int line)
{
    if (error != CUDNN_STATUS_SUCCESS)
        std::cerr << CuDnnError::composeErrorMsg(error, invocation, file, line) << std::endl;
}

void throwCudnnErrorImpl(cudnnStatus_t error, const char *invocation, const char *file, int line)
{
    if (error != CUDNN_STATUS_SUCCESS)
        throw CuDnnError(error, invocation, file, line);
}


}
