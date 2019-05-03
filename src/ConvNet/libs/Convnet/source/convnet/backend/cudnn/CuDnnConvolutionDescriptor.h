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

#ifndef CUDNNCONVOLUTIONDESCRIPTOR_H
#define CUDNNCONVOLUTIONDESCRIPTOR_H

#include <cudnn.h>

#include <vector>

namespace cudnn {

class CuDnnConvolutionDescriptor
{
    public:
        CuDnnConvolutionDescriptor(const CuDnnConvolutionDescriptor&) = delete;
        CuDnnConvolutionDescriptor(const CuDnnConvolutionDescriptor&&) = delete;
        CuDnnConvolutionDescriptor& operator=(const CuDnnConvolutionDescriptor&) = delete;
        CuDnnConvolutionDescriptor& operator=(const CuDnnConvolutionDescriptor&&) = delete;

        CuDnnConvolutionDescriptor();
        ~CuDnnConvolutionDescriptor();

        void setup(const std::vector<unsigned> &padding,
                   const std::vector<unsigned> &stride,
                   const std::vector<unsigned> &upscale,
                   cudnnDataType_t dataType = CUDNN_DATA_FLOAT,
                   cudnnConvolutionMode_t convolutionMode = CUDNN_CONVOLUTION);

        inline cudnnConvolutionDescriptor_t getDescriptor() const { return m_desc; }
    protected:
        cudnnConvolutionDescriptor_t m_desc;
};

}

#endif // CUDNNCONVOLUTIONDESCRIPTOR_H
