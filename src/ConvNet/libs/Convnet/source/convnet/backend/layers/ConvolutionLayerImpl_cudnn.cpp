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

#include "ConvolutionLayerImpl_cudnn.h"

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

#include <iostream>


namespace convnet {


ConvolutionParametersImpl_cudnn::ConvolutionParametersImpl_cudnn(std::string name, ConvNet &convNet, const CuDnnWrapper &cudnnWrapper) :
                ConvolutionParameters(std::move(name)), m_cudnnWrapper(cudnnWrapper),
                m_filters(convNet),
                m_filtersGradient(convNet),
                m_filtersMomentum(convNet),
                m_filtersExpectedGradient(convNet),
                m_bias(convNet),
                m_biasGradient(convNet),
                m_biasMomentum(convNet),
                m_biasExpectedGradient(convNet)
{
}

ConvolutionParametersImpl_cudnn::~ConvolutionParametersImpl_cudnn()
{
}


void ConvolutionParametersImpl_cudnn::structureChanged()
{
    m_filterDesc.setupTightlyPacked({(unsigned) m_kernel.size(), m_filterSize.channels, m_filterSize.depth, m_filterSize.height, m_filterSize.width});
    if (m_hasBias)
        m_biasDesc.setupTightlyPacked({1, (unsigned) m_kernel.size(), 1, 1, 1});

    m_filters.allocate(m_filterSize, m_kernel.size(), DF_FLOAT32);
    m_filtersMomentum.allocate(m_filterSize, m_kernel.size(), DF_FLOAT32);
    m_filtersExpectedGradient.allocate(m_filterSize, m_kernel.size(), DF_FLOAT32);
    m_filtersGradient.allocate(m_filterSize, m_kernel.size(), DF_FLOAT32);

    if (m_hasBias) {
        m_bias.allocate(TensorSize(1, 1, 1, m_kernel.size()), 1, DF_FLOAT32);
        m_biasMomentum.allocate(TensorSize(1, 1, 1, m_kernel.size()), 1, DF_FLOAT32);
        m_biasExpectedGradient.allocate(TensorSize(1, 1, 1, m_kernel.size()), 1, DF_FLOAT32);
        m_biasGradient.allocate(TensorSize(1, 1, 1, m_kernel.size()), 1, DF_FLOAT32);
    }

    m_filters.setZeroSync();
    m_filtersMomentum.setZeroSync();
    m_filtersExpectedGradient.setZeroSync();
    m_filtersGradient.setZeroSync();
    
    if (m_hasBias) {
        m_bias.setZeroSync();
        m_biasMomentum.setZeroSync();
        m_biasExpectedGradient.setZeroSync();
        m_biasGradient.setZeroSync();
    }
    
    if (m_normalization != NORM_NONE)
        throw std::runtime_error("Normalization not implemented yet!");
}


std::size_t ConvolutionParametersImpl_cudnn::getWorkspaceSize() const
{
	return 0;
}


void ConvolutionParametersImpl_cudnn::performParameterStep(float stepsize, ExecutionWorkspace &workspace, ExecutionStream &stream)
{
    ExecutionStreamImpl_cudnn &streamCuImpl = dynamic_cast<ExecutionStreamImpl_cudnn&>(stream);

#if 0
    {
        MappedTensor mappedFilter = m_filtersGradient.lock(false);
        MappedTensor mappedBias = m_biasGradient.lock(false);

        try {
            for (unsigned i = 0; i < m_kernel.size(); i++) {
                std::cout << mappedBias.get<float>(0, 0, 0, i, 0) << std::endl;

                for (unsigned c = 0; c < m_filterSize.channels; c++)
                    for (unsigned z = 0; z < m_filterSize.depth; z++)
                        for (unsigned y = 0; y < m_filterSize.height; y++)
                            for (unsigned x = 0; x < m_filterSize.width; x++)
                                std::cout << mappedFilter.get<float>(x, y, z, c, i) << std::endl;

            }
        } catch (...) {
            m_filters.unlock(mappedFilter, false);
            m_bias.unlock(mappedBias, false);
            throw;
        }
        m_filters.unlock(mappedFilter, false);
        m_bias.unlock(mappedBias, false);
    }
#endif


#if 0
    const CudaUtils::CudaKernel &adaDeltaKernel = m_cudnnWrapper.getAuxKernels().getAdaDeltaKernel();

    {
        AdaDeltaParams params;
        params.values = (float*) m_filters.getDevicePtr();
        params.gradients = (float*) m_filtersGradient.getDevicePtr();
        params.expectedSquaredGradients = (float*) m_filtersExpectedGradient.getDevicePtr();
        params.momentums = (float*) m_filtersMomentum.getDevicePtr();

        params.numElements = m_filters.getTotalNumberOfElements();

        params.stepSize = stepsize;
        params.decay = m_weightDecay;

        adaDeltaKernel.launch({256u, 1u, 1u},
                              {(params.numElements + 256u -1u) / 256u, 1u, 1u},
                              &params,
                              sizeof(params),
                              streamCuImpl.getStream());
    }

    {
        AdaDeltaParams params;
        params.values = (float*) m_bias.getDevicePtr();
        params.gradients = (float*) m_biasGradient.getDevicePtr();
        params.expectedSquaredGradients = (float*) m_biasExpectedGradient.getDevicePtr();
        params.momentums = (float*) m_biasMomentum.getDevicePtr();

        params.numElements = m_bias.getTotalNumberOfElements();

        params.stepSize = stepsize;
        params.decay = m_weightDecay;

        adaDeltaKernel.launch({256u, 1u, 1u},
                              {(params.numElements + 256u -1u) / 256u, 1u, 1u},
                              &params,
                              sizeof(params),
                              streamCuImpl.getStream());
    }
#else
    const CudaUtils::CudaKernel &adamKernel = m_cudnnWrapper.getAuxKernels().getAdamKernel();

    {
        AdamKernelParams params;
        params.values = (float*) m_filters.getDevicePtr();
        params.gradients = (float*) m_filtersGradient.getDevicePtr();
        params.expectedSquaredGradients = (float*) m_filtersExpectedGradient.getDevicePtr();
        params.momentums = (float*) m_filtersMomentum.getDevicePtr();

        params.numElements = m_filters.getTotalNumberOfElements();

        params.stepSize = stepsize * m_weightLearningRate;
        params.decay = m_weightDecay;
        params.lambda1 = m_lambda1;
        params.lambda2 = m_lambda2;
        params.clipMin = m_clipMin;
        params.clipMax = m_clipMax;

        adamKernel.launch({256u, 1u, 1u},
                              {(params.numElements + 256u -1u) / 256u, 1u, 1u},
                              &params,
                              sizeof(params),
                              streamCuImpl.getStream());
    }

    if (m_hasBias) {
        AdamKernelParams params;
        params.values = (float*) m_bias.getDevicePtr();
        params.gradients = (float*) m_biasGradient.getDevicePtr();
        params.expectedSquaredGradients = (float*) m_biasExpectedGradient.getDevicePtr();
        params.momentums = (float*) m_biasMomentum.getDevicePtr();

        params.numElements = m_bias.getTotalNumberOfElements();

        params.stepSize = stepsize * m_biasLearningRate;
        params.decay = m_weightDecay;
        params.lambda1 = m_lambda1;
        params.lambda2 = m_lambda2;
        params.clipMin = -1e30f;
        params.clipMax = 1e30f;

        adamKernel.launch({256u, 1u, 1u},
                              {(params.numElements + 256u -1u) / 256u, 1u, 1u},
                              &params,
                              sizeof(params),
                              streamCuImpl.getStream());
    }

#endif
}

void ConvolutionParametersImpl_cudnn::pushToBackend(ExecutionWorkspace &/*workspace*/, ExecutionStream &stream)
{
    stream.flush();
    MappedTensor mappedFilter = m_filters.lock(true);
    MappedTensor mappedBias;
    if (m_hasBias)
        mappedBias = m_bias.lock(true);

    try {
        for (unsigned i = 0; i < m_kernel.size(); i++) {
            if (m_hasBias)
                mappedBias.get<float>(0, 0, 0, i, 0) = m_kernel[i].bias;

            for (unsigned c = 0; c < m_filterSize.channels; c++)
                for (unsigned z = 0; z < m_filterSize.depth; z++)
                    for (unsigned y = 0; y < m_filterSize.height; y++)
                        for (unsigned x = 0; x < m_filterSize.width; x++)
                            mappedFilter.get<float>(x, y, z, c, i) = m_kernel[i].filter(x, y, z, c, 0);

        }
    } catch (...) {
        m_filters.unlock(mappedFilter);
        if (m_hasBias)
            m_bias.unlock(mappedBias);
        throw;
    }
    m_filters.unlock(mappedFilter);
    if (m_hasBias)
        m_bias.unlock(mappedBias);
}

void ConvolutionParametersImpl_cudnn::pullFromBackend(ExecutionWorkspace &/*workspace*/, ExecutionStream &stream)
{
    stream.flush();
    MappedTensor mappedFilter = m_filters.lock(false);
    MappedTensor mappedBias;
    if (m_hasBias)
        mappedBias = m_bias.lock(false);

    try {
        for (unsigned i = 0; i < m_kernel.size(); i++) {
            if (m_hasBias)
                m_kernel[i].bias = mappedBias.get<float>(0, 0, 0, i, 0);

            for (unsigned c = 0; c < m_filterSize.channels; c++)
                for (unsigned z = 0; z < m_filterSize.depth; z++)
                    for (unsigned y = 0; y < m_filterSize.height; y++)
                        for (unsigned x = 0; x < m_filterSize.width; x++)
                            m_kernel[i].filter(x, y, z, c, 0) = mappedFilter.get<float>(x, y, z, c, i);

        }
    } catch (...) {
        m_filters.unlock(mappedFilter, false);
        if (m_hasBias)
            m_bias.unlock(mappedBias, false);
        throw;
    }
    m_filters.unlock(mappedFilter, false);
    if (m_hasBias)
        m_bias.unlock(mappedBias, false);

#if 0
    {
        MappedTensor mappedFilter = m_filtersExpectedGradient.lock(false);
        MappedTensor mappedBias = m_biasExpectedGradient.lock(false);

        try {
            for (unsigned i = 0; i < m_kernel.size(); i++) {
                std::cout << mappedBias.get<float>(0, 0, 0, i, 0) << std::endl;

                for (unsigned c = 0; c < m_filterSize.channels; c++)
                    for (unsigned z = 0; z < m_filterSize.depth; z++)
                        for (unsigned y = 0; y < m_filterSize.height; y++)
                            for (unsigned x = 0; x < m_filterSize.width; x++)
                                std::cout << mappedFilter.get<float>(x, y, z, c, i) << std::endl;

            }
        } catch (...) {
            m_filters.unlock(mappedFilter, false);
            m_bias.unlock(mappedBias, false);
            throw;
        }
        m_filters.unlock(mappedFilter, false);
        m_bias.unlock(mappedBias, false);
    }
#endif
}

void ConvolutionParametersImpl_cudnn::restoreSnapshot(ExecutionStream &stream, FileBlob &blob)
{
    stream.flush();

    std::uint32_t version, numFilters;

    blob
        >> version
        >> numFilters
        >> m_filterSize
        >> m_weightDecay
        >> m_lambda1
        >> m_lambda2
        >> m_weightLearningRate
        >> m_biasLearningRate;

    m_kernel.resize(numFilters);
    for (auto &k : m_kernel)
        k.filter.allocate(m_filterSize);

    structureChanged();

    auto readTensor = [&blob](TensorImpl_cudnn &tensor) {
        MappedTensor mappedTensor = tensor.lock(true);
        try {
            for (unsigned i = 0; i < mappedTensor.getNumInstances(); i++)
                blob.read(mappedTensor.data<float>(i), mappedTensor.getSize().numElements() * sizeof(float));
        } catch (...) {
            tensor.unlock(mappedTensor, true);
            throw;
        }
        tensor.unlock(mappedTensor, true);
    };

    readTensor(m_filters);
    readTensor(m_filtersMomentum);
    readTensor(m_filtersExpectedGradient);
    readTensor(m_filtersGradient);

    readTensor(m_bias);
    readTensor(m_biasMomentum);
    readTensor(m_biasExpectedGradient);
    readTensor(m_biasGradient);
}

void ConvolutionParametersImpl_cudnn::dumpSnapshot(ExecutionWorkspace &/*workspace*/, ExecutionStream &stream, FileBlob &blob)
{
    stream.flush();

    std::uint32_t version = 1;

    blob
        << version
        << (std::uint32_t) m_kernel.size()
        << m_filterSize
        << m_weightDecay
        << m_lambda1
        << m_lambda2
        << m_weightLearningRate
        << m_biasLearningRate;

    auto dumpTensor = [&blob](TensorImpl_cudnn &tensor) {
        MappedTensor mappedTensor = tensor.lock(false);
        try {
            for (unsigned i = 0; i < mappedTensor.getNumInstances(); i++)
                blob.write(mappedTensor.data<float>(i), mappedTensor.getSize().numElements() * sizeof(float));
        } catch (...) {
            tensor.unlock(mappedTensor, false);
            throw;
        }
        tensor.unlock(mappedTensor, false);
    };

    dumpTensor(m_filters);
    dumpTensor(m_filtersMomentum);
    dumpTensor(m_filtersExpectedGradient);
    dumpTensor(m_filtersGradient);

    dumpTensor(m_bias);
    dumpTensor(m_biasMomentum);
    dumpTensor(m_biasExpectedGradient);
    dumpTensor(m_biasGradient);
}


void ConvolutionParametersImpl_cudnn::printStats()
{
    std::cout << "Stats of " << (std::size_t) this << std::endl;
    const unsigned denomFilter = m_kernel.size() * m_filterSize.numElements();
    const unsigned denomBias = m_kernel.size();
    float avgFilter_expectedGrad = 0.0f;
    float avgBias_expectedGrad = 0.0f;

    {
        MappedTensor mappedFilter = m_filtersExpectedGradient.lock(false);
        MappedTensor mappedBias;
        if (m_hasBias)
            mappedBias = m_biasExpectedGradient.lock(false);

        for (unsigned i = 0; i < m_kernel.size(); i++) {
            if (m_hasBias)
                avgBias_expectedGrad += mappedBias.get<float>(0, 0, 0, i, 0);

            for (unsigned c = 0; c < m_filterSize.channels; c++)
                for (unsigned z = 0; z < m_filterSize.depth; z++)
                    for (unsigned y = 0; y < m_filterSize.height; y++)
                        for (unsigned x = 0; x < m_filterSize.width; x++)
                            avgFilter_expectedGrad += mappedFilter.get<float>(x, y, z, c, i);

        }

        m_filtersExpectedGradient.unlock(mappedFilter, false);
        if (m_hasBias)
            m_biasExpectedGradient.unlock(mappedBias, false);
    }

    avgFilter_expectedGrad /= denomFilter;
    avgBias_expectedGrad /= denomBias;

    float avgFilter_momentum = 0.0f;
    float avgBias_momentum = 0.0f;
    {
        MappedTensor mappedFilter = m_filtersMomentum.lock(false);
        MappedTensor mappedBias;
        if (m_hasBias)
            mappedBias = m_biasMomentum.lock(false);

        for (unsigned i = 0; i < m_kernel.size(); i++) {
            if (m_hasBias)
                avgBias_momentum += std::abs(mappedBias.get<float>(0, 0, 0, i, 0));

            for (unsigned c = 0; c < m_filterSize.channels; c++)
                for (unsigned z = 0; z < m_filterSize.depth; z++)
                    for (unsigned y = 0; y < m_filterSize.height; y++)
                        for (unsigned x = 0; x < m_filterSize.width; x++)
                            avgFilter_momentum += std::abs(mappedFilter.get<float>(x, y, z, c, i));

        }

        m_filtersMomentum.unlock(mappedFilter, false);
        if (m_hasBias)
            m_biasMomentum.unlock(mappedBias, false);
    }

    avgFilter_momentum /= denomFilter;
    avgBias_momentum /= denomBias;

    float avgFilter = 0.0f;
    float avgBias = 0.0f;
    {
        MappedTensor mappedFilter = m_filters.lock(false);
        MappedTensor mappedBias;
        if (m_hasBias)
            mappedBias = m_bias.lock(false);

        for (unsigned i = 0; i < m_kernel.size(); i++) {
            if (m_hasBias)
                avgBias += std::abs(mappedBias.get<float>(0, 0, 0, i, 0));

            for (unsigned c = 0; c < m_filterSize.channels; c++)
                for (unsigned z = 0; z < m_filterSize.depth; z++)
                    for (unsigned y = 0; y < m_filterSize.height; y++)
                        for (unsigned x = 0; x < m_filterSize.width; x++)
                            avgFilter += std::abs(mappedFilter.get<float>(x, y, z, c, i));

        }

        m_filters.unlock(mappedFilter, false);
        if (m_hasBias)
            m_bias.unlock(mappedBias, false);
    }

    avgFilter /= denomFilter;
    avgBias /= denomBias;    
    
    std::cout << "avgFilter_expectedGrad: " << avgFilter_expectedGrad << std::endl;
    std::cout << "avgBias_expectedGrad: " << avgBias_expectedGrad << std::endl;

    std::cout << "avgFilter_momentum: " << avgFilter_momentum << std::endl;
    std::cout << "avgBias_momentum: " << avgBias_momentum << std::endl;

    std::cout << "avgFilter: " << avgFilter << std::endl;
    std::cout << "avgBias: " << avgBias << std::endl;

    std::cout << "avgFilter step w/o stepsize: " << avgFilter_momentum / std::sqrt(1e-10f + avgFilter_expectedGrad) << std::endl;
    std::cout << "avgBias step w/o stepsize: " << avgBias_momentum / std::sqrt(1e-10f + avgBias_expectedGrad) << std::endl;
}



ConvolutionLayerImpl_cudnn::ConvolutionLayerImpl_cudnn(ConvNetImpl_cudnn &convNet, std::string name) :
            ConvolutionLayer(convNet, std::move(name)), m_cudnnWrapper(convNet.getCudnnWrapper())
{
}

void ConvolutionLayerImpl_cudnn::resize(ExecutionStream &stream, Key<ConvNet> key)
{
    ConvolutionLayer::resize(stream, key);

    m_convDesc.setup({(unsigned)m_padding[2], (unsigned)m_padding[1], (unsigned)m_padding[0]},
                     {(unsigned)m_stride[2], (unsigned)m_stride[1], (unsigned)m_stride[0]},
                     {(unsigned)m_upsample[2], (unsigned)m_upsample[1], (unsigned)m_upsample[0]});

    ExecutionStreamImpl_cudnn &streamCuImpl = dynamic_cast<ExecutionStreamImpl_cudnn&>(stream);

    ConvolutionParametersImpl_cudnn *params = dynamic_cast<ConvolutionParametersImpl_cudnn*>(m_parameters);

    const TensorSize &inputSize = m_connectionList->outputs[m_inputs[0]].size;
    TensorSize &outputSize = m_connectionList->outputs[m_outputs[0]].size;


    m_inputTensorDesc.setupTightlyPacked({m_convnet.getNumInstances(), inputSize.channels, inputSize.depth, inputSize.height, inputSize.width});
    m_outputTensorDesc.setupTightlyPacked({m_convnet.getNumInstances(), outputSize.channels, outputSize.depth, outputSize.height, outputSize.width});

    {
        int dims[5];
        cudnnGetConvolutionNdForwardOutputDim(
            m_convDesc.getDescriptor(),
            m_inputTensorDesc.getDescriptor(),
            params->m_filterDesc.getDescriptor(),
            5,
            dims
        );

        std::cout
            << dims[0] <<  " "
            << dims[1] <<  " "
            << dims[2] <<  " "
            << dims[3] <<  " "
            << dims[4] <<  std::endl;
        std::cout
            << m_convnet.getNumInstances() <<  " "
            << outputSize.channels <<  " "
            << outputSize.depth <<  " "
            << outputSize.height <<  " "
            << outputSize.width <<  std::endl;

        if (dims[0] != (int)m_convnet.getNumInstances())
            throw std::runtime_error("Internal Error 0!");
        if (dims[1] != (int)outputSize.channels)
            throw std::runtime_error("Internal Error 1!");
        if (dims[2] != (int)outputSize.depth)
            throw std::runtime_error("Internal Error 2!");
        if (dims[3] != (int)outputSize.height)
            throw std::runtime_error("Internal Error 3!");
        if (dims[4] != (int)outputSize.width)
            throw std::runtime_error("Internal Error 4!");
    }



    m_workspaceMemory = 0;

    {
        int count;
        cudnnConvolutionFwdAlgoPerf_t algos[10];
        throwCudnnError(cudnnFindConvolutionForwardAlgorithm(
                            streamCuImpl.getContext().getHandle(),
                            m_inputTensorDesc.getDescriptor(),
                            params->m_filterDesc.getDescriptor(),
                            m_convDesc.getDescriptor(),
                            m_outputTensorDesc.getDescriptor(),
                            10,
                            &count,
                            algos
        ));
        /*
        std::cout << "Found " << count << " algos: " << std::endl;
        for (unsigned i = 0; i < (unsigned)count; i++) {
            std::cout << "algo: " << algos[i].algo << "   status: " << cudnnGetErrorString(algos[i].status) << "   time: " << algos[i].time << "   memory: " << algos[i].memory << std::endl;
        }
        */
        m_forwardAlgorithm = algos[0].algo;
        m_workspaceMemory = std::max(m_workspaceMemory, algos[0].memory);
    }


    {
        int count;
        cudnnConvolutionBwdDataAlgoPerf_t algos[10];
        throwCudnnError(cudnnFindConvolutionBackwardDataAlgorithm(
                            streamCuImpl.getContext().getHandle(),
                            params->m_filterDesc.getDescriptor(),
                            m_outputTensorDesc.getDescriptor(),
                            m_convDesc.getDescriptor(),
                            m_inputTensorDesc.getDescriptor(),
                            10,
                            &count,
                            algos
        ));
        /*
        std::cout << "Found " << count << " algos: " << std::endl;
        for (unsigned i = 0; i < (unsigned)count; i++) {
            std::cout << "algo: " << algos[i].algo << "   status: " << cudnnGetErrorString(algos[i].status) << "   time: " << algos[i].time << "   memory: " << algos[i].memory << std::endl;
        }
        */
        m_backwardAlgorithm = algos[0].algo;
        m_workspaceMemory = std::max(m_workspaceMemory, algos[0].memory);
    }

    {
        int count;
        cudnnConvolutionBwdFilterAlgoPerf_t algos[10];
        throwCudnnError(cudnnFindConvolutionBackwardFilterAlgorithm(
                            streamCuImpl.getContext().getHandle(),
                            m_inputTensorDesc.getDescriptor(),
                            m_outputTensorDesc.getDescriptor(),
                            m_convDesc.getDescriptor(),
                            params->m_filterDesc.getDescriptor(),
                            10,
                            &count,
                            algos
        ));
        /*
        std::cout << "Found " << count << " algos: " << std::endl;
        for (unsigned i = 0; i < (unsigned)count; i++) {
            std::cout << "algo: " << algos[i].algo << "   status: " << cudnnGetErrorString(algos[i].status) << "   time: " << algos[i].time << "   memory: " << algos[i].memory << std::endl;
        }
        */
        m_backwardFilterAlgorithm = algos[0].algo;
        m_workspaceMemory = std::max(m_workspaceMemory, algos[0].memory);
    }
    m_workspaceMemory = std::max(m_workspaceMemory, params->getWorkspaceSize());
}

void ConvolutionLayerImpl_cudnn::allocateState(NetworkState &networkState, ExecutionStream &/*stream*/, bool forward, bool backward, bool parameterUpdate, Key<ConvNet>) const
{
    for (unsigned i : m_outputs) {
        TensorDataImpl_cudnn *data;
        networkState.outputs[i].reset(data = new TensorDataImpl_cudnn(m_convnet));
        data->allocate(m_connectionList->outputs[i].size, m_convnet.getNumInstances(), DF_FLOAT32, backward);
    }
}


void ConvolutionLayerImpl_cudnn::forward(NetworkState &networkState, ExecutionWorkspace &workspace, ExecutionStream &stream, bool /*trainingMode*/) const
{
    TensorDataImpl_cudnn &input = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[0]]);
    TensorDataImpl_cudnn &output = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_outputs[0]]);
    ConvolutionParametersImpl_cudnn *params = dynamic_cast<ConvolutionParametersImpl_cudnn*>(m_parameters);
    ExecutionStreamImpl_cudnn &streamCuImpl = dynamic_cast<ExecutionStreamImpl_cudnn&>(stream);
    ExecutionWorkspaceImpl_cudnn &workspaceCuImpl = dynamic_cast<ExecutionWorkspaceImpl_cudnn&>(workspace);


    float one = 1.0f;
    float zero = 0.0f;

    throwCudnnError(cudnnConvolutionForward(
        streamCuImpl.getContext().getHandle(),
        &one,
        m_inputTensorDesc.getDescriptor(),
        input.getValuesDevicePtr(),
        params->m_filterDesc.getDescriptor(),
        params->m_filters.getDevicePtr(),
        m_convDesc.getDescriptor(),
        m_forwardAlgorithm,
        workspaceCuImpl.getWorkspaceMemory().getPtr(),
        workspaceCuImpl.getWorkspaceMemory().size(),
        &zero,
        m_outputTensorDesc.getDescriptor(),
        output.getValuesDevicePtr()
    ));

    if (params->m_hasBias)
        throwCudnnError(cudnnAddTensor(
            streamCuImpl.getContext().getHandle(),
            &one,
            params->m_biasDesc.getDescriptor(),
            params->m_bias.getDevicePtr(),
            &one,
            m_outputTensorDesc.getDescriptor(),
            output.getValuesDevicePtr()
        ));
}

void ConvolutionLayerImpl_cudnn::backward(NetworkState &networkState, ExecutionWorkspace &workspace, ExecutionStream &stream, bool accumulateParameterGradients) const
{
    TensorDataImpl_cudnn &input = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_inputs[0]]);
    TensorDataImpl_cudnn &output = dynamic_cast<TensorDataImpl_cudnn&>(*networkState.outputs[m_outputs[0]]);
    ConvolutionParametersImpl_cudnn *params = dynamic_cast<ConvolutionParametersImpl_cudnn*>(m_parameters);
    ExecutionStreamImpl_cudnn &streamCuImpl = dynamic_cast<ExecutionStreamImpl_cudnn&>(stream);
    ExecutionWorkspaceImpl_cudnn &workspaceCuImpl = dynamic_cast<ExecutionWorkspaceImpl_cudnn&>(workspace);

    float one = 1.0f;

    throwCudnnError(cudnnConvolutionBackwardData(
        streamCuImpl.getContext().getHandle(),
        &one,
        params->m_filterDesc.getDescriptor(),
        params->m_filters.getDevicePtr(),
        m_outputTensorDesc.getDescriptor(),
        output.getGradientsDevicePtr(),
        m_convDesc.getDescriptor(),
        m_backwardAlgorithm,
        workspaceCuImpl.getWorkspaceMemory().getPtr(),
        workspaceCuImpl.getWorkspaceMemory().size(),
        &one,
        m_inputTensorDesc.getDescriptor(),
        input.getGradientsDevicePtr()
    ));

    if (accumulateParameterGradients) {
        throwCudnnError(cudnnConvolutionBackwardFilter(
            streamCuImpl.getContext().getHandle(),
            &one,
            m_inputTensorDesc.getDescriptor(),
            input.getValuesDevicePtr(),
            m_outputTensorDesc.getDescriptor(),
            output.getGradientsDevicePtr(),
            m_convDesc.getDescriptor(),
            m_backwardFilterAlgorithm,
            workspaceCuImpl.getWorkspaceMemory().getPtr(),
            workspaceCuImpl.getWorkspaceMemory().size(),
            &one,
            params->m_filterDesc.getDescriptor(),
            params->m_filtersGradient.getDevicePtr()
        ));

        if (params->m_hasBias)
            throwCudnnError(cudnnConvolutionBackwardBias(
                streamCuImpl.getContext().getHandle(),
                &one,
                m_outputTensorDesc.getDescriptor(),
                output.getGradientsDevicePtr(),
                &one,
                params->m_biasDesc.getDescriptor(),
                params->m_biasGradient.getDevicePtr()
            ));
    }
}

ConvolutionParameters *ConvolutionLayerImpl_cudnn::instantiateParameters(std::string name)
{
    return new ConvolutionParametersImpl_cudnn(std::move(name), m_convnet, m_cudnnWrapper);
}

}
