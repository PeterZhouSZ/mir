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

#ifndef NETWORKSTATE_H
#define NETWORKSTATE_H

#include "Tensor.h"

#include <memory>
#include <vector>

namespace convnet {

class ConvNet;

class NetworkData {
    public:
        NetworkData(ConvNet &/*convnet*/) { }
        virtual ~NetworkData() = default;
        virtual void clear(ExecutionStream &stream) = 0;

};

class TensorData : public NetworkData
{
    public:
        TensorData(ConvNet &convnet) : NetworkData(convnet) { }
        virtual Tensor &getValues() = 0;
        virtual Tensor &getGradients() = 0;
        virtual void clear(ExecutionStream &stream) override;
        void allocate(const TensorSize &size, unsigned numInstances, DataFormat format, bool gradients = true);
};


class AuxiliaryNetworkLayerData {
    public:
        virtual ~AuxiliaryNetworkLayerData() = default;
};

struct NetworkState
{
    std::vector<std::unique_ptr<NetworkData>> outputs;
    std::vector<std::unique_ptr<AuxiliaryNetworkLayerData>> auxLayerData;
    void clear(ExecutionStream &stream);
};

struct InputStagingBuffer
{
    std::vector<std::unique_ptr<NetworkData>> outputs;
};

}

#endif // NETWORKSTATE_H
