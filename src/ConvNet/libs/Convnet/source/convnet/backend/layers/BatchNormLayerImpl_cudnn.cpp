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

#include "BatchNormLayerImpl_cudnn.h"

#include "../ConvNetImpl_cudnn.h"
#include "../ExecutionWorkspaceImpl_cudnn.h"
#include "../ExecutionStreamImpl_cudnn.h"

#include "../CuDnnWrapper.h"

#include "../cudnn/CuDnnError.h"

#include "../../NetworkState.h"
#include "../../FileBlob.h"

//#include "../cudaKernels/AdaDeltaKernel.h"
#include "../cudaKernels/AdamKernel.h"

#include <cudnn.h>


namespace convnet {


BatchNormParametersImpl_cudnn::BatchNormParametersImpl_cudnn(std::string name, ConvNet &convNet, const CuDnnWrapper &cudnnWrapper) :
                BatchNormParameters(std::move(name)), m_cudnnWrapper(cudnnWrapper),
                m_runningMean(convNet),
                m_runningVar(convNet),
                m_scale(convNet),
                m_scaleGradient(convNet),
                m_scaleMomentum(convNet),
                m_scaleExpectedGradient(convNet),
                m_bias(convNet),
                m_biasGradient(convNet),
                m_biasMomentum(convNet),
                m_biasExpectedGradient(convNet)
{
}

BatchNormParametersImpl_cudnn::~BatchNormParametersImpl_cudnn()
{
}


void BatchNormParametersImpl_cudnn::structureChanged()
{
    m_runningMean.allocate(TensorSize(1, 1, 1, m_channels.size()), 1, DF_FLOAT32);
    m_runningVar.allocate(TensorSize(1, 1, 1, m_channels.size()), 1, DF_FLOAT32);
    
    m_scale.allocate(TensorSize(1, 1, 1, m_channels.size()), 1, DF_FLOAT32);
    m_scaleGradient.allocate(TensorSize(1, 1, 1, m_channels.size()), 1, DF_FLOAT32);
    m_scaleMomentum.allocate(TensorSize(1, 1, 1, m_channels.size()), 1, DF_FLOAT32);
    m_scaleExpectedGradient.allocate(TensorSize(1, 1, 1, m_channels.size()), 1, DF_FLOAT32);
    
    m_bias.allocate(TensorSize(1, 1, 1, m_channels.size()), 1, DF_FLOAT32);
    m_biasGradient.allocate(TensorSize(1, 1, 1, m_channels.size()), 1, DF_FLOAT32);
    m_biasMomentum.allocate(TensorSize(1, 1, 1, m_channels.size()), 1, DF_FLOAT32);
    m_biasExpectedGradient.allocate(TensorSize(1, 1, 1, m_channels.size()), 1, DF_FLOAT32);  
    
    m_tensorDesc.setupTightlyPacked({1, m_channels.size(), 1, 1, 1});
    
    m_runningMean.setZeroSync();
    m_runningVar.setZeroSync();
    
    m_scale.setZeroSync();
    m_scaleGradient.setZeroSync();
    m_scaleMomentum.setZeroSync();
    m_scaleExpectedGradient.setZeroSync();
    
    m_bias.setZeroSync();
    m_biasMomentum.setZeroSync();
    m_biasExpectedGradient.setZeroSync();
    m_biasGradient.setZeroSync();
}


void BatchNormParametersImpl_cudnn::performParameterStep(float stepsize, ExecutionWorkspace &/*workspace*/, ExecutionStream &stream)
{
    ExecutionStreamImpl_cudnn &streamCuImpl = dynamic_cast<ExecutionStreamImpl_cudnn&>(stream);

    const CudaUtils::CudaKernel &adamKernel = m_cudnnWrapper.getAuxKernels().getAdamKernel();

    if (m_adaptBias) {
        AdamKernelParams params;
        params.values = (float*) m_bias.getDevicePtr();
        params.gradients = (float*) m_biasGradient.getDevicePtr();
        params.expectedSquaredGradients = (float*) m_biasExpectedGradient.getDevicePtr();
        params.momentums = (float*) m_biasMomentum.getDevicePtr();

        params.numElements = m_bias.getTotalNumberOfElements();

        params.stepSize = stepsize * m_biasLearningRate;
        params.decay = 0.0f;
        params.lambda1 = m_lambda1;
        params.lambda2 = m_lambda2;
        params.clipMin = -1e30f;
        params.clipMax = 1e30f;

        adamKernel.launch(
            {32u, 1u, 1u},
            {(params.numElements + 32u -1u) / 32u, 1u, 1u},
            &params,
            sizeof(params),
            streamCuImpl.getStream()
        );
    }
    
    if (m_adaptScale) {
        AdamKernelParams params;
        params.values = (float*) m_scale.getDevicePtr();
        params.gradients = (float*) m_scaleGradient.getDevicePtr();
        params.expectedSquaredGradients = (float*) m_scaleExpectedGradient.getDevicePtr();
        params.momentums = (float*) m_scaleMomentum.getDevicePtr();

        params.numElements = m_scale.getTotalNumberOfElements();

        params.stepSize = stepsize * m_scaleLearningRate;
        params.decay = 0.0f;
        params.lambda1 = m_lambda1;
        params.lambda2 = m_lambda2;
        params.clipMin = -1e30f;
        params.clipMax = 1e30f;

        adamKernel.launch(
            {32u, 1u, 1u},
            {(params.numElements + 32u -1u) / 32u, 1u, 1u},
            &params,
            sizeof(params),
            streamCuImpl.getStream()
        );
    }
}

void BatchNormParametersImpl_cudnn::pushToBackend(ExecutionWorkspace &/*workspace*/, ExecutionStream &stream)
{
    stream.flush();
    MappedTensor mappedScale = m_scale.lock(true);
    MappedTensor mappedBias = m_bias.lock(true);
    MappedTensor mappedFirstMoment = m_runningMean.lock(true);
    MappedTensor mappedSecondMoment = m_runningVar.lock(true);

        for (unsigned i = 0; i < m_channels.size(); i++) {
            mappedBias.get<float>(0, 0, 0, i, 0) = m_channels[i].bias;
            mappedScale.get<float>(0, 0, 0, i, 0) = m_channels[i].scale;
            mappedFirstMoment.get<float>(0, 0, 0, i, 0) = m_channels[i].runningMean;
            mappedSecondMoment.get<float>(0, 0, 0, i, 0) = m_channels[i].runningVar;
        }
    m_scale.unlock(mappedScale);
    m_bias.unlock(mappedBias);
    m_runningMean.unlock(mappedFirstMoment);
    m_runningVar.unlock(mappedSecondMoment);
}

void BatchNormParametersImpl_cudnn::pullFromBackend(ExecutionWorkspace &/*workspace*/, ExecutionStream &stream)
{
    stream.flush();
    MappedTensor mappedScale = m_scale.lock(false);
    MappedTensor mappedBias = m_bias.lock(false);
    MappedTensor mappedFirstMoment = m_runningMean.lock(false);
    MappedTensor mappedSecondMoment = m_runningVar.lock(false);

        for (unsigned i = 0; i < m_channels.size(); i++) {
            m_channels[i].bias = mappedBias.get<float>(0, 0, 0, i, 0);
            m_channels[i].scale = mappedScale.get<float>(0, 0, 0, i, 0);
            m_channels[i].runningMean = mappedFirstMoment.get<float>(0, 0, 0, i, 0);
            m_channels[i].runningVar = mappedSecondMoment.get<float>(0, 0, 0, i, 0);
        }
    m_scale.unlock(mappedScale, false);
    m_bias.unlock(mappedBias, false);
    m_runningMean.unlock(mappedFirstMoment, false);
    m_runningVar.unlock(mappedSecondMoment, false);
}


BatchNormLayerImpl_cudnn_AuxData::BatchNormLayerImpl_cudnn_AuxData(ConvNet &convNet, unsigned numChannels) :
    m_mean(convNet),
    m_invVar(convNet)
{
    m_mean.allocate(TensorSize(1, 1, 1, numChannels), 1, DF_FLOAT32);
    m_invVar.allocate(TensorSize(1, 1, 1, numChannels), 1, DF_FLOAT32);
}


BatchNormLayerImpl_cudnn::BatchNormLayerImpl_cudnn(ConvNetImpl_cudnn &convNet, std::string name) :
            BatchNormLayer(convNet, std::move(name)), m_cudnnWrapper(convNet.getCudnnWrapper())
{
}

void BatchNormLayerImpl_cudnn::resize(ExecutionStream &stream, Key<ConvNet> key)
{
    BatchNormLayer::resize(stream, key);
    
    //BatchNormParametersImpl_cudnn *params = dynamic_cast<BatchNormParametersImpl_cudnn*>(m_parameters);

    const TensorSize &inputSize = m_connectionList->outputs[m_inputs[0]].size;
    const TensorSize &outputSize = m_connectionList->outputs[m_outputs[0]].size;

    m_inputTensorDesc.setupTightlyPacked({m_convnet.getNumInstances(), inputSize.channels, inputSize.depth, inputSize.height, inputSize.width});
    m_outputTensorDesc.setupTightlyPacked({m_convnet.getNumInstances(), outputSize.channels, outputSize.depth, outputSize.height, outputSize.width});
}

void BatchNormLayerImpl_cudnn::allocateState(NetworkState &networkState, ExecutionStream &/*stream*/, bool forward, bool backward, bool parameterUpdate, Key<ConvNet>) const
{
    for (unsigned i : m_outputs) {
        TensorDataImpl_cudnn *data;
        networkState.outputs[i].reset(data = new TensorDataImpl_cudnn(m_convnet));
        data->allocate(m_connectionList->outputs[i].size, m_convnet.getNumInstances(), DF_FLOAT32, backward);
    }

    networkState.auxLayerData[m_layerIndex].reset(new BatchNormLayerImpl_cudnn_AuxData(
        m_convnet,
        m_connectionList->outputs[m_inputs[0]].size.channels
    ));
}

void BatchNormLayerImpl_cudnn::forward(NetworkState &networkState, ExecutionWorkspace &workspace, ExecutionStream &stream, bool trainingMode) const
{
    TensorDataImpl_cudnn &input = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[0]]);
    TensorDataImpl_cudnn &output = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_outputs[0]]);
    BatchNormParametersImpl_cudnn *parameters = dynamic_cast<BatchNormParametersImpl_cudnn*>(m_parameters);
    ExecutionStreamImpl_cudnn &streamCuImpl = dynamic_cast<ExecutionStreamImpl_cudnn&>(stream);
    ExecutionWorkspaceImpl_cudnn &workspaceCuImpl = dynamic_cast<ExecutionWorkspaceImpl_cudnn&>(workspace);
    BatchNormLayerImpl_cudnn_AuxData &auxData = dynamic_cast<BatchNormLayerImpl_cudnn_AuxData&>(*networkState.auxLayerData[m_layerIndex]);
    
//trainingMode = true;
    
    auxData.m_trainingMode = trainingMode;
    
    float zero = 0.0f;
    float one = 1.0f;
    
    const CudnnAuxKernels &auxKernel = m_cudnnWrapper.getAuxKernels();
    if (trainingMode) {
        throwCudnnError(cudnnBatchNormalizationForwardTraining(
            streamCuImpl.getContext().getHandle(),
            CUDNN_BATCHNORM_SPATIAL,
            &one,
            &zero,
            m_inputTensorDesc.getDescriptor(),
            input.getValuesDevicePtr(),
            m_outputTensorDesc.getDescriptor(),
            output.getValuesDevicePtr(),
            parameters->m_tensorDesc.getDescriptor(),
            parameters->m_scale.getDevicePtr(),
            parameters->m_bias.getDevicePtr(),
            1.0f - m_parameters->m_momentLambda,
            parameters->m_runningMean.getDevicePtr(),
            parameters->m_runningVar.getDevicePtr(),
            parameters->epsilon,
            auxData.m_mean.getDevicePtr(),
            auxData.m_invVar.getDevicePtr()
        ));
    } else {
        throwCudnnError(cudnnBatchNormalizationForwardInference(
            streamCuImpl.getContext().getHandle(),
            CUDNN_BATCHNORM_SPATIAL,
            &one,
            &zero,
            m_inputTensorDesc.getDescriptor(),
            input.getValuesDevicePtr(),
            m_outputTensorDesc.getDescriptor(),
            output.getValuesDevicePtr(),
            parameters->m_tensorDesc.getDescriptor(),
            parameters->m_scale.getDevicePtr(),
            parameters->m_bias.getDevicePtr(),
            parameters->m_runningMean.getDevicePtr(),
            parameters->m_runningVar.getDevicePtr(),
            parameters->epsilon
        ));
    }
}

void BatchNormLayerImpl_cudnn::backward(NetworkState &networkState, ExecutionWorkspace &workspace, ExecutionStream &stream, bool accumulateParameterGradients) const
{
    TensorDataImpl_cudnn &input = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[0]]);
    TensorDataImpl_cudnn &output = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_outputs[0]]);
    BatchNormParametersImpl_cudnn *parameters = dynamic_cast<BatchNormParametersImpl_cudnn*>(m_parameters);
    ExecutionStreamImpl_cudnn &streamCuImpl = dynamic_cast<ExecutionStreamImpl_cudnn&>(stream);
    ExecutionWorkspaceImpl_cudnn &workspaceCuImpl = dynamic_cast<ExecutionWorkspaceImpl_cudnn&>(workspace);
    BatchNormLayerImpl_cudnn_AuxData &auxData = dynamic_cast<BatchNormLayerImpl_cudnn_AuxData&>(*networkState.auxLayerData[m_layerIndex]);
    
    //const CudnnAuxKernels &auxKernel = m_cudnnWrapper.getAuxKernels();
    
    float one = 1.0f;
    float zero = 0.0f;
    
    if (auxData.m_trainingMode || m_backpropAlwaysTrainigMode) {
        throwCudnnError(cudnnBatchNormalizationBackward(
            streamCuImpl.getContext().getHandle(),
            CUDNN_BATCHNORM_SPATIAL,
            &one,
            &one,
            accumulateParameterGradients?&one:&zero,
            &one,
            m_inputTensorDesc.getDescriptor(),
            input.getValuesDevicePtr(),
            m_outputTensorDesc.getDescriptor(),
            output.getGradientsDevicePtr(),
            m_inputTensorDesc.getDescriptor(),
            input.getGradientsDevicePtr(),
            parameters->m_tensorDesc.getDescriptor(),
            parameters->m_scale.getDevicePtr(),
            parameters->m_scaleGradient.getDevicePtr(),
            parameters->m_biasGradient.getDevicePtr(),
            parameters->epsilon,
            auxData.m_trainingMode?auxData.m_mean.getDevicePtr():nullptr,
            auxData.m_trainingMode?auxData.m_invVar.getDevicePtr():nullptr
        ));
    } else {
      
    }
}

BatchNormParameters *BatchNormLayerImpl_cudnn::instantiateParameters(std::string name)
{
    return new BatchNormParametersImpl_cudnn(std::move(name), m_convnet, m_cudnnWrapper);
}

}
