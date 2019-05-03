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

struct ScalarCrossEntropyLossForwardParams {
    float *inputValues;
    float *inputLabels;
    float *loss;
    unsigned numLabels;
    float scale;
};

struct ScalarCrossEntropyLossBackwardParams {
    float *inputValues;
    float *inputLabels;
    float *gradients;
    unsigned numLabels;
    float scale;
};

inline unsigned ScalarCrossEntropyLoss_forwardSharedMemorySize(unsigned numLabels, unsigned threadsPerBlock)
{
    return (numLabels * threadsPerBlock) * sizeof(float);
}

inline unsigned ScalarCrossEntropyLoss_backwardSharedMemorySize(unsigned numLabels, unsigned threadsPerBlock)
{
    return (numLabels * threadsPerBlock) * sizeof(float);
}

struct CrossEntropyLossForwardParams {
    float *inputInferredLabels;
    float *inputGTLabels;
    float *loss;
    float scale;

    unsigned width;
    unsigned height;
    unsigned depth;

    unsigned strideX[3];
    unsigned strideY[3];
    unsigned strideZ[3];
    unsigned strideC[1];
    unsigned strideN[3];
    
    unsigned numInstances;
    unsigned numLabels;
};

struct CrossEntropyLossBackwardParams {
    float *inputInferredLabels;
    float *inputGTLabels;
    float *gradients;
    float scale;

    unsigned width;
    unsigned height;
    unsigned depth;

    unsigned strideX[2];
    unsigned strideY[2];
    unsigned strideZ[2];
    unsigned strideC[1];
    unsigned strideN[2];
    
    unsigned numInstances;
    unsigned numLabels;
};

inline unsigned CrossEntropyLoss_forwardSharedMemorySize(unsigned numLabels, unsigned threadsPerBlock)
{
    return (numLabels * threadsPerBlock) * sizeof(float);
}

inline unsigned CrossEntropyLoss_backwardSharedMemorySize(unsigned numLabels, unsigned threadsPerBlock)
{
    return (numLabels * threadsPerBlock) * sizeof(float);
}

