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

#include "StridedCopyKernel.h"

extern "C" __global__ void stridedCopyKernel(StridedCopyKernelParams kernelParams)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned c = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= kernelParams.dims[0]) return;
    if (y >= kernelParams.dims[1]) return;
    if (c >= kernelParams.dims[3]) return;
    
    float *output = kernelParams.outputValues 
                        + x 
                        + y * kernelParams.stridesOutput[0] 
                        + c * kernelParams.stridesOutput[2];
    
    const float *input = kernelParams.inputValues 
                        + x 
                        + y * kernelParams.stridesInput[0] 
                        + c * kernelParams.stridesInput[2];

    for (unsigned i = 0; i < kernelParams.dims[4]; i++) {
        for (unsigned z = 0; z < kernelParams.dims[2]; z++) {
            output[z * kernelParams.stridesOutput[1] + i * kernelParams.stridesOutput[3]] = 
                input[z * kernelParams.stridesInput[1] + i * kernelParams.stridesInput[3]];
        }
    }
}

extern "C" __global__ void stridedCopyAddKernel(StridedCopyKernelParams kernelParams)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned c = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= kernelParams.dims[0]) return;
    if (y >= kernelParams.dims[1]) return;
    if (c >= kernelParams.dims[3]) return;
    
    float *output = kernelParams.outputValues 
                        + x 
                        + y * kernelParams.stridesOutput[0] 
                        + c * kernelParams.stridesOutput[2];
    
    const float *input = kernelParams.inputValues 
                        + x 
                        + y * kernelParams.stridesInput[0] 
                        + c * kernelParams.stridesInput[2];

    for (unsigned i = 0; i < kernelParams.dims[4]; i++) {
        for (unsigned z = 0; z < kernelParams.dims[2]; z++) {
            output[z * kernelParams.stridesOutput[1] + i * kernelParams.stridesOutput[3]] += 
                input[z * kernelParams.stridesInput[1] + i * kernelParams.stridesInput[3]];
        }
    }
}
