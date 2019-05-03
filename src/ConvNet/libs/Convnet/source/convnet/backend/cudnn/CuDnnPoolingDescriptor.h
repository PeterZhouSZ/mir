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

#ifndef CUDNNPOOLINGDESCRIPTOR_H
#define CUDNNPOOLINGDESCRIPTOR_H

#include <cudnn.h>

#include <vector>

namespace cudnn {

class CuDnnPoolingDescriptor
{
    public:
        CuDnnPoolingDescriptor(const CuDnnPoolingDescriptor&) = delete;
        CuDnnPoolingDescriptor(const CuDnnPoolingDescriptor&&) = delete;
        CuDnnPoolingDescriptor& operator=(const CuDnnPoolingDescriptor&) = delete;
        CuDnnPoolingDescriptor& operator=(const CuDnnPoolingDescriptor&&) = delete;

        CuDnnPoolingDescriptor();
        ~CuDnnPoolingDescriptor();

        void setup(const std::vector<unsigned> &window, const std::vector<unsigned> &padding, const std::vector<unsigned> &stride,
                   cudnnPoolingMode_t mode, cudnnNanPropagation_t maxPoolNanProp = CUDNN_PROPAGATE_NAN);

        inline cudnnPoolingDescriptor_t getDescriptor() const { return m_desc; }
    protected:
        cudnnPoolingDescriptor_t m_desc;

};

}

#endif // CUDNNPOOLINGDESCRIPTOR_H
