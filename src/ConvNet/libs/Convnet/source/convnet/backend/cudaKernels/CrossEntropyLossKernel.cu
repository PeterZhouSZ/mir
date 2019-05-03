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

#include "CrossEntropyLossKernel.h"


extern "C" __global__ void scalarCrossEntropyLossForwardKernel(ScalarCrossEntropyLossForwardParams kernelParams)
{
    const unsigned instance = threadIdx.x;

    const float *values = kernelParams.inputValues + instance * kernelParams.numLabels;
    const float *label = kernelParams.inputLabels + instance;
    float *loss = kernelParams.loss + instance;

    unsigned l = (unsigned) *label;
    if (l == 0xFF) {
        *loss = 0.0f;
        return;
    }
    
    if (l >= kernelParams.numLabels) {
        *loss = 0.0f;
        printf("Invalid label: %i\n", l);
        return;
    }

    float *buffer = sharedMemory + threadIdx.x;
    const unsigned bufferStride = blockDim.x;


    float maxOut = buffer[0] = values[0];
    for (unsigned c = 0; c < kernelParams.numLabels; c++) {
        float f = values[c];
        buffer[c*bufferStride] = f;
        maxOut = fmaxf(maxOut, f);
    }

    float denom = 0.0f;
    for (unsigned c = 0; c < kernelParams.numLabels; c++) {
        float f = __expf(buffer[c*bufferStride] - maxOut);
        denom += f;
    }

    *loss = (__logf(denom) - (buffer[l*bufferStride] - maxOut)) * kernelParams.scale;
}


extern "C" __global__ void scalarCrossEntropyLossBackwardKernel(ScalarCrossEntropyLossBackwardParams kernelParams)
{
    const unsigned instance = threadIdx.x;

    const float *values = kernelParams.inputValues + instance * kernelParams.numLabels;
    const float *label = kernelParams.inputLabels + instance;
    float *gradients = kernelParams.gradients + instance * kernelParams.numLabels;

    unsigned l = (unsigned) *label;
    if (l == 0xFF) return;


    float *buffer = sharedMemory + threadIdx.x;
    const unsigned bufferStride = blockDim.x;


    float maxOut = buffer[0] = values[0];
    for (unsigned c = 0; c < kernelParams.numLabels; c++) {
        float f = values[c];
        buffer[c*bufferStride] = f;
        maxOut = fmaxf(maxOut, f);
    }

    float denom = 0.0f;
    for (unsigned c = 0; c < kernelParams.numLabels; c++) {
        float f = __expf(buffer[c*bufferStride] - maxOut);
        denom += f;
        buffer[c*bufferStride] = f;
    }

    for (unsigned c = 0; c <  kernelParams.numLabels; c++) {
        float f = buffer[c*bufferStride] / denom;
        if (c == l)
            f -= 1.0f;

        gradients[c] += f * kernelParams.scale;
    }
}


extern "C" __global__ void crossEntropyLossForwardKernel(CrossEntropyLossForwardParams kernelParams)
{
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned x = index % kernelParams.width;
    const unsigned y = (index / kernelParams.width) % kernelParams.height;
    const unsigned z = (index / kernelParams.width / kernelParams.height) % kernelParams.depth;    
    const unsigned instance = index / kernelParams.width / kernelParams.height / kernelParams.depth;
    
    if (instance >= kernelParams.numInstances) return;

    const float *inferredLabel = kernelParams.inputInferredLabels
                                    + x * kernelParams.strideX[0]
                                    + y * kernelParams.strideY[0]
                                    + z * kernelParams.strideZ[0]
                                    + instance * kernelParams.strideN[0];
                                    
    const float *GTLabel = kernelParams.inputGTLabels
                                    + x * kernelParams.strideX[1]
                                    + y * kernelParams.strideY[1]
                                    + z * kernelParams.strideZ[1]
                                    + instance * kernelParams.strideN[1];

    float *loss = kernelParams.loss
                                    + x * kernelParams.strideX[2]
                                    + y * kernelParams.strideY[2]
                                    + z * kernelParams.strideZ[2]
                                    + instance * kernelParams.strideN[2];

    unsigned l = (unsigned) *GTLabel;
    if (l == 0xFF) {
        *loss = 0.0f;
        return;
    }

    if (l >= kernelParams.numLabels) {
        *loss = 0.0f;
        printf("Invalid label: %i\n", l);
        return;
    }
    
    float *buffer = sharedMemory + threadIdx.x;
    const unsigned bufferStride = blockDim.x;

    float maxOut = buffer[0] = inferredLabel[0];
    for (unsigned c = 1; c < kernelParams.numLabels; c++) {
        float f = inferredLabel[c*kernelParams.strideC[0]];
        buffer[c*bufferStride] = f;
        maxOut = fmaxf(maxOut, f);
    }

    float denom = 0.0f;
    for (unsigned c = 0; c < kernelParams.numLabels; c++) {
        float f = __expf(buffer[c*bufferStride] - maxOut);
        denom += f;
    }

    *loss = (__logf(denom) - (buffer[l*bufferStride] - maxOut)) * kernelParams.scale;
}

extern "C" __global__ void crossEntropyLossBackwardKernel(CrossEntropyLossBackwardParams kernelParams)
{
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned x = index % kernelParams.width;
    const unsigned y = (index / kernelParams.width) % kernelParams.height;
    const unsigned z = (index / kernelParams.width / kernelParams.height) % kernelParams.depth;    
    const unsigned instance = index / kernelParams.width / kernelParams.height / kernelParams.depth;
    
    if (instance >= kernelParams.numInstances) return;

    const float *inferredLabel = kernelParams.inputInferredLabels
                                    + x * kernelParams.strideX[0]
                                    + y * kernelParams.strideY[0]
                                    + z * kernelParams.strideZ[0]
                                    + instance * kernelParams.strideN[0];
                                    
    const float *GTLabel = kernelParams.inputGTLabels
                                    + x * kernelParams.strideX[1]
                                    + y * kernelParams.strideY[1]
                                    + z * kernelParams.strideZ[1]
                                    + instance * kernelParams.strideN[1];

    float *gradients = kernelParams.gradients
                                    + x * kernelParams.strideX[0]
                                    + y * kernelParams.strideY[0]
                                    + z * kernelParams.strideZ[0]
                                    + instance * kernelParams.strideN[0];
    unsigned l = (unsigned) *GTLabel;
    if (l == 0xFF) return;

    float *buffer = sharedMemory + threadIdx.x;
    const unsigned bufferStride = blockDim.x;

    float maxOut = buffer[0] = inferredLabel[0];
    for (unsigned c = 1; c < kernelParams.numLabels; c++) {
        float f = inferredLabel[c*kernelParams.strideC[0]];
        buffer[c*bufferStride] = f;
        maxOut = fmaxf(maxOut, f);
    }

    float denom = 0.0f;
    for (unsigned c = 0; c < kernelParams.numLabels; c++) {
        float f = __expf(buffer[c*bufferStride] - maxOut);
        denom += f;
        buffer[c*bufferStride] = f;
    }

    for (unsigned c = 0; c <  kernelParams.numLabels; c++) {
        float f = buffer[c*bufferStride] / denom;
        if (c == l)
            f -= 1.0f;

        gradients[c*kernelParams.strideC[0]] += f * kernelParams.scale;
    }
}
