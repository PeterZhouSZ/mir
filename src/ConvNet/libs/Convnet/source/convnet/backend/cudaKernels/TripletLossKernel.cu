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

#include "TripletLossKernel.h"

#if 1
extern "C" __global__ void TripletLossForwardKernel(TripletLossForwardParams kernelParams)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= kernelParams.widthHeightDepth) return;
    
    unsigned instance = blockIdx.y;

    float loss = kernelParams.margin;
    for (unsigned c = 0; c < kernelParams.channels; c++) {
        float anchor = kernelParams.inputAnchor[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN];
        float match = kernelParams.inputMatch[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN];
        float nonMatch = kernelParams.inputNonMatch[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN];

        loss += (anchor - match)*(anchor - match) - (anchor - nonMatch)*(anchor - nonMatch);        
    }

    kernelParams.loss[x + instance * kernelParams.widthHeightDepth] = fmaxf(loss * kernelParams.weight * 0.5f, 0.0f);
}


extern "C" __global__ void TripletLossBackwardKernel(TripletLossBackwardParams kernelParams)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= kernelParams.widthHeightDepth) return;
    
    unsigned instance = blockIdx.y;

    float loss = kernelParams.loss[x + instance * kernelParams.widthHeightDepth];
    if (loss == 0.0f) return;
    
    for (unsigned c = 0; c < kernelParams.channels; c++) {
        float anchor = kernelParams.inputAnchor[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN];
        float match = kernelParams.inputMatch[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN];
        float nonMatch = kernelParams.inputNonMatch[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN];

        kernelParams.gradientAnchor[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN] += (nonMatch - match) * kernelParams.weight;
        kernelParams.gradientMatch[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN] += (match - anchor) * kernelParams.weight;
        kernelParams.gradientNonMatch[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN] += (anchor - nonMatch) * kernelParams.weight;
    }
}
#else
extern "C" __global__ void TripletLossForwardKernel(TripletLossForwardParams kernelParams)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= kernelParams.widthHeightDepth) return;
    
    unsigned instance = blockIdx.y;

    float loss1 = 0.0f;
    float loss2 = 0.0f;
    for (unsigned c = 0; c < kernelParams.channels; c++) {
        float anchor = kernelParams.inputAnchor[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN];
        float match = kernelParams.inputMatch[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN];
        float nonMatch = kernelParams.inputNonMatch[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN];

        loss1 += (anchor - match)*(anchor - match);
		loss2 += (anchor - nonMatch)*(anchor - nonMatch);        
    }

    kernelParams.loss[x + instance * kernelParams.widthHeightDepth] = (fmaxf(loss1 - kernelParams.margin, 0.0f) + fmaxf(2*kernelParams.margin - loss2, 0.0f)) * kernelParams.weight * 0.5f;
}


extern "C" __global__ void TripletLossBackwardKernel(TripletLossBackwardParams kernelParams)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= kernelParams.widthHeightDepth) return;
    
    unsigned instance = blockIdx.y;

    float loss1 = 0.0f;
    float loss2 = 0.0f;
    for (unsigned c = 0; c < kernelParams.channels; c++) {
        float anchor = kernelParams.inputAnchor[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN];
        float match = kernelParams.inputMatch[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN];
        float nonMatch = kernelParams.inputNonMatch[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN];

        loss1 += (anchor - match)*(anchor - match);
		loss2 += (anchor - nonMatch)*(anchor - nonMatch);        
    }
    
    for (unsigned c = 0; c < kernelParams.channels; c++) {
        float anchor = kernelParams.inputAnchor[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN];
        float match = kernelParams.inputMatch[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN];
        float nonMatch = kernelParams.inputNonMatch[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN];

		if (loss1 > kernelParams.margin) {
	        kernelParams.gradientAnchor[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN] += (anchor - match) * kernelParams.weight;
    	    kernelParams.gradientMatch[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN] += (match - anchor) * kernelParams.weight;
		}
		if (loss2 < 2*kernelParams.margin) {
	        kernelParams.gradientAnchor[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN] += (nonMatch - anchor) * kernelParams.weight;
	        kernelParams.gradientNonMatch[c * kernelParams.widthHeightDepth + x + instance * kernelParams.strideN] += (anchor - nonMatch) * kernelParams.weight;
		}
    }
}
#endif
