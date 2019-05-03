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

#ifndef CUDNNAUXKERNELS_H
#define CUDNNAUXKERNELS_H

#include <cudaUtils/CudaCodeModule.h>
#include <cudaUtils/CudaKernel.h>

#include <memory>

namespace convnet {

class CudnnAuxKernels
{
    public:
        void loadKernels();
/*
        inline const CudaUtils::CudaKernel &getPReluForwardKernel() const { return *m_preluForwardKernel; }
        inline const CudaUtils::CudaKernel &getPReluBackwardKernel() const { return *m_preluBackwardKernel; }
        inline const CudaUtils::CudaKernel &getPReluBackwardPGradKernel() const { return *m_preluBackwardPGradKernel; }

        inline const CudaUtils::CudaKernel &getReluSplitForwardKernel() const { return *m_reluSplitForwardKernel; }
        inline const CudaUtils::CudaKernel &getReluSplitBackwardKernel() const { return *m_reluSplitBackwardKernel; }
        inline const CudaUtils::CudaKernel &getReluSplitGuidedBackwardKernel() const { return *m_reluSplitGuidedBackwardKernel; }

        inline const CudaUtils::CudaKernel &getUnpoolingForwardKernel() const { return *m_unpoolingForwardKernel; }
        inline const CudaUtils::CudaKernel &getUnpoolingBackwardKernel() const { return *m_unpoolingBackwardKernel; }

        inline const CudaUtils::CudaKernel &getVAEBottleneckForwardKernel() const { return *m_VAEBottleneckForwardKernel; }
        inline const CudaUtils::CudaKernel &getVAEBottleneckBackwardKernel() const { return *m_VAEBottleneckBackwardKernel; }

        inline const CudaUtils::CudaKernel &getConcatenationForwardKernel() const { return *m_concatenationForwardKernel; }
        inline const CudaUtils::CudaKernel &getConcatenationBackwardKernel() const { return *m_concatenationBackwardKernel; }
*/
        inline const CudaUtils::CudaKernel &getCrossEntropyLossForwardKernel(bool scalar) const { return *m_crossEntropyLossForwardKernel[scalar?0:1]; }
        inline const CudaUtils::CudaKernel &getCrossEntropyLossBackwardKernel(bool scalar) const { return *m_crossEntropyLossBackwardKernel[scalar?0:1]; }
/*
        inline const CudaUtils::CudaKernel &getReconstructionLossForwardKernel(unsigned lossType) const { return *m_reconstructionLossForwardKernel[lossType]; }
        inline const CudaUtils::CudaKernel &getReconstructionLossBackwardKernel(unsigned lossType) const { return *m_reconstructionLossBackwardKernel[lossType]; }

        inline const CudaUtils::CudaKernel &getElemWiseOpForwardKernel(unsigned opType) const { return *m_elemWiseOpForwardKernel[opType]; }
        inline const CudaUtils::CudaKernel &getElemWiseOpBackwardKernel(unsigned opType) const { return *m_elemWiseOpBackwardKernel[opType]; }

        inline const CudaUtils::CudaKernel &getAdaDeltaKernel() const { return *m_adaDeltaKernel; }
*/
        inline const CudaUtils::CudaKernel &getAdamKernel() const { return *m_adamKernel; }
/*
        inline const CudaUtils::CudaKernel &getNegateTensorKernel() const { return *m_negateTensorKernel; }

        inline const CudaUtils::CudaKernel &getGANLossForwardKernel(bool descriminator, bool noSoftmax) const { return *m_ganLossForwardKernel[(descriminator?0:1)+(noSoftmax?0:2)]; }
        inline const CudaUtils::CudaKernel &getGANLossBackwardKernel(bool descriminator, bool noSoftmax) const { return *m_ganLossBackwardKernel[(descriminator?0:1)+(noSoftmax?0:2)]; }

        inline const CudaUtils::CudaKernel &getSaturationForwardKernel(unsigned type) const { return *m_saturationForwardKernel[type]; }
        inline const CudaUtils::CudaKernel &getSaturationBackwardKernel(unsigned type) const { return *m_saturationBackwardKernel[type]; }
*/
        inline const CudaUtils::CudaKernel &getTripletLossForwardKernel() const { return *m_tripletLossForwardKernel; }
        inline const CudaUtils::CudaKernel &getTripletLossBackwardKernel() const { return *m_tripletLossBackwardKernel; }
        
        
        inline const CudaUtils::CudaKernel &getInplaceSumKernel() const { return *m_inplaceSumKernel; }
        
        inline const CudaUtils::CudaKernel &getStridedCopyKernel() const { return *m_stridedCopyKernel; }
        inline const CudaUtils::CudaKernel &getStridedCopyAddKernel() const { return *m_stridedCopyAddKernel; }
        inline const CudaUtils::CudaKernel &getStridedMemsetKernel() const { return *m_stridedMemsetKernel; }
        /*
        enum BatchRenormKernel {
            BATCH_RENORM_SUM_MOMENTS,
            BATCH_RENORM_HANDLE_BATCH_MOMENTS,
            BATCH_RENORM_COMPUTE_BATCH_FACTORS,
            BATCH_RENORM_FORWARD,
            BATCH_RENORM_COMPUTENORMALIZATION_FACTORS,
            BATCH_RENORM_BACKWARD_STEP1,
            BATCH_RENORM_BACKWARD_STEP2,
            BATCH_RENORM_BACKWARD_STEP3,
            BATCH_RENORM_BACKWARD_INFERENCE,
            BATCH_RENORM_UPDATE_RUNNING_AVG,
            BATCH_RENORM_NUM_KERNELS
        };
        
        inline const CudaUtils::CudaKernel &getBatchRenormKernel(BatchRenormKernel kernel) const { return *m_batchRenormKernel[kernel]; }
        */
    protected:
        CudaUtils::CudaCodeModule m_codeModule;
/*
        std::unique_ptr<CudaUtils::CudaKernel> m_preluForwardKernel;
        std::unique_ptr<CudaUtils::CudaKernel> m_preluBackwardKernel;
        std::unique_ptr<CudaUtils::CudaKernel> m_preluBackwardPGradKernel;

        std::unique_ptr<CudaUtils::CudaKernel> m_reluSplitForwardKernel;
        std::unique_ptr<CudaUtils::CudaKernel> m_reluSplitBackwardKernel;
        std::unique_ptr<CudaUtils::CudaKernel> m_reluSplitGuidedBackwardKernel;

        std::unique_ptr<CudaUtils::CudaKernel> m_unpoolingForwardKernel;
        std::unique_ptr<CudaUtils::CudaKernel> m_unpoolingBackwardKernel;

        std::unique_ptr<CudaUtils::CudaKernel> m_VAEBottleneckForwardKernel;
        std::unique_ptr<CudaUtils::CudaKernel> m_VAEBottleneckBackwardKernel;

        std::unique_ptr<CudaUtils::CudaKernel> m_concatenationForwardKernel;
        std::unique_ptr<CudaUtils::CudaKernel> m_concatenationBackwardKernel;
*/
        std::unique_ptr<CudaUtils::CudaKernel> m_crossEntropyLossForwardKernel[2];
        std::unique_ptr<CudaUtils::CudaKernel> m_crossEntropyLossBackwardKernel[2];
/*
        std::unique_ptr<CudaUtils::CudaKernel> m_reconstructionLossForwardKernel[3];
        std::unique_ptr<CudaUtils::CudaKernel> m_reconstructionLossBackwardKernel[3];
        
        std::unique_ptr<CudaUtils::CudaKernel> m_adaDeltaKernel;
*/
        std::unique_ptr<CudaUtils::CudaKernel> m_adamKernel;
/*
        std::unique_ptr<CudaUtils::CudaKernel> m_elemWiseOpForwardKernel[1];
        std::unique_ptr<CudaUtils::CudaKernel> m_elemWiseOpBackwardKernel[1];

        std::unique_ptr<CudaUtils::CudaKernel> m_negateTensorKernel;

        std::unique_ptr<CudaUtils::CudaKernel> m_ganLossForwardKernel[2*2];
        std::unique_ptr<CudaUtils::CudaKernel> m_ganLossBackwardKernel[2*2];

        std::unique_ptr<CudaUtils::CudaKernel> m_saturationForwardKernel[1];
        std::unique_ptr<CudaUtils::CudaKernel> m_saturationBackwardKernel[1];
*/
        std::unique_ptr<CudaUtils::CudaKernel> m_tripletLossForwardKernel;
        std::unique_ptr<CudaUtils::CudaKernel> m_tripletLossBackwardKernel;

        std::unique_ptr<CudaUtils::CudaKernel> m_stridedCopyKernel;
        std::unique_ptr<CudaUtils::CudaKernel> m_stridedCopyAddKernel;        
        std::unique_ptr<CudaUtils::CudaKernel> m_stridedMemsetKernel;        
        
        std::unique_ptr<CudaUtils::CudaKernel> m_inplaceSumKernel;
        
//        std::unique_ptr<CudaUtils::CudaKernel> m_batchRenormKernel[BATCH_RENORM_NUM_KERNELS];
};

}

#endif // CUDNNAUXKERNELS_H
