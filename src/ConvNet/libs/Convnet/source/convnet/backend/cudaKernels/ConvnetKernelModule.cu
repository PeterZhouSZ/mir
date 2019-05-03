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

extern __shared__ float sharedMemory[];

#ifndef __shfl_xor_sync
#define __shfl_xor_sync __shfl_xor
#endif

//#include "PReluKernel.cu"
//#include "ReluSplitKernel.cu"
//#include "UnpoolingKernel.cu"
//#include "AdaDeltaKernel.cu"
//#include "VAEBottleneckKernel.cu"
//#include "ConcatenationKernel.cu"
#include "CrossEntropyLossKernel.cu"
//#include "ReconstructionLossKernel.cu"
#include "AdamKernel.cu"
//#include "ElemWiseOpKernel.cu"
//#include "NegateTensorKernel.cu"
//#include "GANLossKernel.cu"
//#include "SaturationKernel.cu"
//#include "BatchRenormKernel.cu"
#include "TripletLossKernel.cu"

#include "StridedCopyKernel.cu"
#include "StridedMemsetKernel.cu"

#include "MiscKernel.cu"
