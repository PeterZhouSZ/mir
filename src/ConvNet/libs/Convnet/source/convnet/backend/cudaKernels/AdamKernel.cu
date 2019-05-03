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

#include "AdamKernel.h"


extern "C" __global__ void AdamKernel(AdamKernelParams kernelParams)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kernelParams.numElements) return;

#if 0
    const float lambda1 = 0.995f;
    const float lambda2 = 0.995f;
#else
    const float lambda1 = kernelParams.lambda1;
    const float lambda2 = kernelParams.lambda2;
#endif

    float value = kernelParams.values[index];
    float gradient = kernelParams.gradients[index];
    /*
    if (index < 10)
        printf("Gradient: %f\n", gradient);
    */
    gradient += value * kernelParams.decay;

    float expectedSquaredGradient = kernelParams.expectedSquaredGradients[index];

    /*
    if (kernelParams.stepSize != 0.0f) { // not burnin phase
        if (fabs(gradient) > sqrtf(expectedSquaredGradient) * 3.0f)
            gradient = copysignf(sqrtf(expectedSquaredGradient) * 3.0f, gradient);
    }
    */

    expectedSquaredGradient = expectedSquaredGradient * lambda1 + gradient*gradient * (1.0f - lambda1);

    if (kernelParams.stepSize != 0.0f) { // not burnin phase
        float momentum = kernelParams.momentums[index];
        momentum = momentum * lambda2 + gradient * (1.0f - lambda2);


        // skipping bias correction

        float update = kernelParams.stepSize * momentum / sqrtf(1e-16f + expectedSquaredGradient);
        update = fminf(fmaxf(update, -0.01f), 0.01f);

        value += update;

        kernelParams.values[index] = fminf(fmaxf(value, kernelParams.clipMin), kernelParams.clipMax);
        kernelParams.momentums[index] = momentum;
    }

    kernelParams.gradients[index] = 0.0f;
    kernelParams.expectedSquaredGradients[index] = expectedSquaredGradient;
}

