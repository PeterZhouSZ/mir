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

#ifndef CUDNNACTIVATIONDESCRIPTOR_H
#define CUDNNACTIVATIONDESCRIPTOR_H

#include <cudnn.h>

namespace cudnn {

class CuDnnActivationDescriptor
{
    public:
        CuDnnActivationDescriptor(const CuDnnActivationDescriptor&) = delete;
        CuDnnActivationDescriptor(const CuDnnActivationDescriptor&&) = delete;
        CuDnnActivationDescriptor& operator=(const CuDnnActivationDescriptor&) = delete;
        CuDnnActivationDescriptor& operator=(const CuDnnActivationDescriptor&&) = delete;

        CuDnnActivationDescriptor();
        ~CuDnnActivationDescriptor();

        void setup(cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU, cudnnNanPropagation_t reluNanOpt = CUDNN_PROPAGATE_NAN, double reluCeiling = 2.0f);

        inline cudnnActivationDescriptor_t getDescriptor() const { return m_desc; }
    protected:
        cudnnActivationDescriptor_t m_desc;
};

}

#endif // CUDNNACTIVATIONDESCRIPTOR_H
