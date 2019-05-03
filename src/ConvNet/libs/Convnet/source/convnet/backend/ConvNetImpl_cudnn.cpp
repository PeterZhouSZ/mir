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

#include "ConvNetImpl_cudnn.h"

#include "layers/ConvolutionLayerImpl_cudnn.h"
//#include "layers/TransposedConvolutionLayerImpl_cudnn.h"
//#include "layers/ReluSplitLayerImpl_cudnn.h"
#include "layers/PoolingLayerImpl_cudnn.h"
//#include "layers/UnpoolingLayerImpl_cudnn.h"
#include "layers/InputLayerImpl_cudnn.h"
//#include "layers/VAEBottleneckLayerImpl_cudnn.h"
//#include "layers/ConcatenationLayerImpl_cudnn.h"
#include "layers/ReluLayerImpl_cudnn.h"
//#include "layers/PReluLayerImpl_cudnn.h"
#include "layers/DropoutLayerImpl_cudnn.h"
//#include "layers/CrossEntropyLossLayerImpl_cudnn.h"
//#include "layers/ReconstructionLossLayerImpl_cudnn.h"
//#include "layers/ElemWiseOpLayerImpl_cudnn.h"
//#include "layers/GradientNegatingLayerImpl_cudnn.h"
//#include "layers/GANLossLayerImpl_cudnn.h"
//#include "layers/SaturationLayerImpl_cudnn.h"
//#include "layers/LocalResponseNormalizationLayerImpl_cudnn.h"
//#include "layers/BatchRenormLayerImpl_cudnn.h"
//#include "layers/BatchNormLayerImpl_cudnn.h"
//#include "layers/CropLayerImpl_cudnn.h"
#include "layers/TripletLossLayerImpl_cudnn.h"

#include "ExecutionWorkspaceImpl_cudnn.h"
#include "ExecutionStreamImpl_cudnn.h"

#include <stdexcept>

namespace convnet {


ConvNetImpl_cudnn::ConvNetImpl_cudnn(CuDnnWrapper &cudnnWrapper) : m_cudnnWrapper(cudnnWrapper)
{
}


std::unique_ptr<ExecutionWorkspace> ConvNetImpl_cudnn::allocateExecutionWorkspace(bool /*forward*/, bool /*backward*/, bool /*parameterUpdate*/)
{
    std::size_t workspaceMemory = 0;
    for (auto &l : m_layers)
        workspaceMemory = std::max(workspaceMemory, l->getWorkspaceSize());

    ExecutionWorkspaceImpl_cudnn *ec = new ExecutionWorkspaceImpl_cudnn(workspaceMemory);
    std::unique_ptr<ExecutionWorkspace> result(ec);
    return result;
}

std::unique_ptr<ExecutionStream> ConvNetImpl_cudnn::allocateExecutionStream()
{
    return std::unique_ptr<ExecutionStream>(new ExecutionStreamImpl_cudnn());
}

std::unique_ptr<ExecutionStreamWaitingFence> ConvNetImpl_cudnn::allocateWaitFence()
{
    return std::unique_ptr<ExecutionStreamWaitingFence>(new ExecutionStreamWaitingFenceImpl_cudnn());
}

std::unique_ptr<ExecutionStreamSyncFence> ConvNetImpl_cudnn::allocateSyncFence()
{
    return std::unique_ptr<ExecutionStreamSyncFence>(new ExecutionStreamSyncFenceImpl_cudnn());
}



Layer *ConvNetImpl_cudnn::instantiateLayer(std::string name, const std::string &type)
{
    if (type == ConvolutionLayer::TypeStr)
        return new ConvolutionLayerImpl_cudnn(*this, std::move(name));
/*
    if (type == TransposedConvolutionLayer::TypeStr)
        return new TransposedConvolutionLayerImpl_cudnn(*this, std::move(name));
    if (type == ReluSplitLayer::TypeStr)
        return new ReluSplitLayerImpl_cudnn(*this, std::move(name));
*/
    if (type == PoolingLayer::TypeStr)
        return new PoolingLayerImpl_cudnn(*this, std::move(name));
/*
    if (type == UnpoolingLayer::TypeStr)
        return new UnpoolingLayerImpl_cudnn(*this, std::move(name));
*/
    if (type == InputLayer::TypeStr)
        return new InputLayerImpl_cudnn(*this, std::move(name));
/*
    if (type == VAEBottleneckLayer::TypeStr)
        return new VAEBottleneckLayerImpl_cudnn(*this, std::move(name));
    if (type == ConcatenationLayer::TypeStr)
        return new ConcatenationLayerImpl_cudnn(*this, std::move(name));
*/
    if (type == ReluLayer::TypeStr)
        return new ReluLayerImpl_cudnn(*this, std::move(name));
/*
    if (type == PReluLayer::TypeStr)
        return new PReluLayerImpl_cudnn(*this, std::move(name));
*/
    if (type == DropoutLayer::TypeStr)
        return new DropoutLayerImpl_cudnn(*this, std::move(name));
/*
    if (type == CrossEntropyLossLayer::TypeStr)
        return new CrossEntropyLossLayerImpl_cudnn(*this, std::move(name));
    if (type == ReconstructionLossLayer::TypeStr)
        return new ReconstructionLossLayerImpl_cudnn(*this, std::move(name));
    if (type == ElemWiseOpLayer::TypeStr)
        return new ElemWiseOpLayerImpl_cudnn(*this, std::move(name));
    if (type == GradientNegatingLayer::TypeStr)
        return new GradientNegatingLayerImpl_cudnn(*this, std::move(name));
    if (type == GANLossLayer::TypeStr)
        return new GANLossLayerImpl_cudnn(*this, std::move(name));
    if (type == SaturationLayer::TypeStr)
        return new SaturationLayerImpl_cudnn(*this, std::move(name));
    if (type == LocalResponseNormalizationLayer::TypeStr)
        return new LocalResponseNormalizationLayerImpl_cudnn(*this, std::move(name));
    if (type == CropLayer::TypeStr)
        return new CropLayerImpl_cudnn(*this, std::move(name));
    if (type == BatchRenormLayerImpl_cudnn::TypeStr)
        return new BatchRenormLayerImpl_cudnn(*this, std::move(name));
    if (type == BatchNormLayerImpl_cudnn::TypeStr)
        return new BatchNormLayerImpl_cudnn(*this, std::move(name));
*/
    if (type == TripletLossLayer::TypeStr)
        return new TripletLossLayerImpl_cudnn(*this, std::move(name));

    throw std::runtime_error(std::string("Found no implementation for layer ") + type);

    return nullptr;
}

}
