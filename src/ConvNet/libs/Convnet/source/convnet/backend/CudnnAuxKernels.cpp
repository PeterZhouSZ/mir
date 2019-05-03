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

#include "CudnnAuxKernels.h"


extern const unsigned char PTX_ConvnetKernelModule[];

namespace convnet {

void CudnnAuxKernels::loadKernels()
{
    m_codeModule.loadFromMemory(PTX_ConvnetKernelModule);

/*
    m_preluForwardKernel.reset(m_codeModule.getKernel("preluForwardKernel"));
    m_preluBackwardKernel.reset(m_codeModule.getKernel("preluBackwardKernel"));
    m_preluBackwardPGradKernel.reset(m_codeModule.getKernel("preluBackwardPGradKernel"));

    m_reluSplitForwardKernel.reset(m_codeModule.getKernel("reluSplitForwardKernel"));
    m_reluSplitBackwardKernel.reset(m_codeModule.getKernel("reluSplitBackwardKernel"));
    m_reluSplitGuidedBackwardKernel.reset(m_codeModule.getKernel("reluSplitGuidedBackwardKernel"));

    m_unpoolingForwardKernel.reset(m_codeModule.getKernel("unpoolingForwardKernel"));
    m_unpoolingBackwardKernel.reset(m_codeModule.getKernel("unpoolingBackwardKernel"));

    m_VAEBottleneckForwardKernel.reset(m_codeModule.getKernel("VAEBottleneckForwardKernel"));
    m_VAEBottleneckBackwardKernel.reset(m_codeModule.getKernel("VAEBottleneckBackwardKernel"));

    m_concatenationForwardKernel.reset(m_codeModule.getKernel("concatenationForwardKernel"));
    m_concatenationBackwardKernel.reset(m_codeModule.getKernel("concatenationBackwardKernel"));
*/
    m_crossEntropyLossForwardKernel[0].reset(m_codeModule.getKernel("scalarCrossEntropyLossForwardKernel"));
    m_crossEntropyLossBackwardKernel[0].reset(m_codeModule.getKernel("scalarCrossEntropyLossBackwardKernel"));
    m_crossEntropyLossForwardKernel[1].reset(m_codeModule.getKernel("crossEntropyLossForwardKernel"));
    m_crossEntropyLossBackwardKernel[1].reset(m_codeModule.getKernel("crossEntropyLossBackwardKernel"));
/*
    m_reconstructionLossForwardKernel[0].reset(m_codeModule.getKernel("reconstructionLossForwardKernel_L1_2"));
    m_reconstructionLossBackwardKernel[0].reset(m_codeModule.getKernel("reconstructionLossBackwardKernel_L1_2"));
    m_reconstructionLossForwardKernel[1].reset(m_codeModule.getKernel("reconstructionLossForwardKernel_L1"));
    m_reconstructionLossBackwardKernel[1].reset(m_codeModule.getKernel("reconstructionLossBackwardKernel_L1"));
    m_reconstructionLossForwardKernel[2].reset(m_codeModule.getKernel("reconstructionLossForwardKernel_L2"));
    m_reconstructionLossBackwardKernel[2].reset(m_codeModule.getKernel("reconstructionLossBackwardKernel_L2"));

    m_elemWiseOpForwardKernel[0].reset(m_codeModule.getKernel("elemWiseOpForwardKernel"));
    m_elemWiseOpBackwardKernel[0].reset(m_codeModule.getKernel("elemWiseOpBackwardKernel"));


    m_adaDeltaKernel.reset(m_codeModule.getKernel("AdaDeltaKernel"));
*/
    m_adamKernel.reset(m_codeModule.getKernel("AdamKernel"));
/*
    m_negateTensorKernel.reset(m_codeModule.getKernel("NegateTensorKernel"));

    m_ganLossForwardKernel[0].reset(m_codeModule.getKernel("ganDescriminatorLossNoSoftmaxForwardKernel"));
    m_ganLossBackwardKernel[0].reset(m_codeModule.getKernel("ganDescriminatorLossNoSoftmaxBackwardKernel"));

    m_ganLossForwardKernel[1].reset(m_codeModule.getKernel("ganGeneratorLossNoSoftmaxForwardKernel"));
    m_ganLossBackwardKernel[1].reset(m_codeModule.getKernel("ganGeneratorLossNoSoftmaxBackwardKernel"));

    m_ganLossForwardKernel[2].reset(m_codeModule.getKernel("ganDescriminatorLossForwardKernel"));
    m_ganLossBackwardKernel[2].reset(m_codeModule.getKernel("ganDescriminatorLossBackwardKernel"));

    m_ganLossForwardKernel[3].reset(m_codeModule.getKernel("ganGeneratorLossForwardKernel"));
    m_ganLossBackwardKernel[3].reset(m_codeModule.getKernel("ganGeneratorLossBackwardKernel"));

    m_saturationForwardKernel[0].reset(m_codeModule.getKernel("SaturationTanhForwardKernel"));
    m_saturationBackwardKernel[0].reset(m_codeModule.getKernel("SaturationTanhBackwardKernel"));
*/
    m_tripletLossForwardKernel.reset(m_codeModule.getKernel("TripletLossForwardKernel"));
    m_tripletLossBackwardKernel.reset(m_codeModule.getKernel("TripletLossBackwardKernel"));
    
    m_stridedCopyKernel.reset(m_codeModule.getKernel("stridedCopyKernel"));
    m_stridedCopyAddKernel.reset(m_codeModule.getKernel("stridedCopyAddKernel"));
    m_stridedMemsetKernel.reset(m_codeModule.getKernel("stridedMemsetKernel"));
    
    m_inplaceSumKernel.reset(m_codeModule.getKernel("InplaceSumKernel"));
/*
    m_batchRenormKernel[BATCH_RENORM_SUM_MOMENTS].reset(m_codeModule.getKernel("batchRenormSumBatchMomentsKernel"));
    m_batchRenormKernel[BATCH_RENORM_HANDLE_BATCH_MOMENTS].reset(m_codeModule.getKernel("batchRenormHandleBatchMomentsKernel"));
    m_batchRenormKernel[BATCH_RENORM_COMPUTE_BATCH_FACTORS].reset(m_codeModule.getKernel("batchRenormComputeBatchFactorsKernel"));
    m_batchRenormKernel[BATCH_RENORM_FORWARD].reset(m_codeModule.getKernel("batchRenormForwardKernel"));
    m_batchRenormKernel[BATCH_RENORM_COMPUTENORMALIZATION_FACTORS].reset(m_codeModule.getKernel("batchRenormComputeNormalizationFactorsKernel"));
    m_batchRenormKernel[BATCH_RENORM_BACKWARD_STEP1].reset(m_codeModule.getKernel("batchRenormBackwardStep1Kernel"));
    m_batchRenormKernel[BATCH_RENORM_BACKWARD_STEP2].reset(m_codeModule.getKernel("batchRenormBackwardStep2Kernel"));
    m_batchRenormKernel[BATCH_RENORM_BACKWARD_STEP3].reset(m_codeModule.getKernel("batchRenormBackwardStep3Kernel"));
    m_batchRenormKernel[BATCH_RENORM_BACKWARD_INFERENCE].reset(m_codeModule.getKernel("batchRenormBackwardInferenceKernel"));
    m_batchRenormKernel[BATCH_RENORM_UPDATE_RUNNING_AVG].reset(m_codeModule.getKernel("batchRenormUpdateRunningAvgKernel"));
*/
}


}
