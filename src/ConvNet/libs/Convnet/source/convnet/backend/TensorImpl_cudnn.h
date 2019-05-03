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

#ifndef TENSORIMPL_CUDNN_H
#define TENSORIMPL_CUDNN_H

#include "../Tensor.h"
#include "../NetworkState.h"

#include <cudaUtils/CudaDeviceMemory.h>
#include <cudaUtils/CudaStream.h>

namespace convnet {

class TensorImpl_cudnn : public Tensor
{
    public:
        TensorImpl_cudnn(ConvNet &convnet) : Tensor(convnet) { }

        virtual void allocate(const TensorSize &size, unsigned numInstances, DataFormat format) override;

        virtual MappedTensor lock(bool discard = false) override;
        virtual void unlock(MappedTensor mappedTensor, bool dirty = true) override;

        virtual void setZero(ExecutionStream &stream) override;
        virtual void setZeroSync() override;

        void* getDevicePtr() { return m_deviceMemory.getPtr(); }
        std::size_t getMemoryAmount() const { return m_deviceMemory.size(); }
    protected:
        CudaUtils::CudaDeviceMemory m_deviceMemory;

};

class TensorDataImpl_cudnn : public TensorData
{
    public:
        TensorDataImpl_cudnn(ConvNet &convnet) : TensorData(convnet), values(convnet), gradients(convnet) { }

        virtual Tensor &getValues() override { return values; }
        virtual Tensor &getGradients() override { return gradients; }

        void* getValuesDevicePtr() { return values.getDevicePtr(); }
        void* getGradientsDevicePtr() { return gradients.getDevicePtr(); }
    protected:
        TensorImpl_cudnn values;
        TensorImpl_cudnn gradients;
};

}

#endif // TENSORIMPL_CUDNN_H
