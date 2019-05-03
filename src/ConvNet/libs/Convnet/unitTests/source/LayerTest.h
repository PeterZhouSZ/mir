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

#ifndef LAYERTEST_H
#define LAYERTEST_H

#include <convnet/Convnet.h>
#include <convnet/NetworkState.h>
#include <convnet/ExecutionWorkspace.h>
#include <convnet/ExecutionStream.h>




class DataDerivativeTest {
    public:
        DataDerivativeTest(convnet::ConvNet &net, unsigned batchSize);
        void testInput(unsigned inputIndex, unsigned x, unsigned y, unsigned z, unsigned c, unsigned n);
        void testAll();
    protected:
        convnet::ConvNet &m_net;
        convnet::NetworkState m_states[3];
        std::unique_ptr<convnet::ExecutionWorkspace> m_executionWorkspace;
        std::unique_ptr<convnet::ExecutionStream> m_executionStream;
};

class LossLayerTest {
    public:
        LossLayerTest(convnet::ConvNet &net, unsigned batchSize, std::function<float(unsigned)> produceValue);
        void testInput(unsigned inputIndex, unsigned x, unsigned y, unsigned z, unsigned c, unsigned n);
        void testAll();
    protected:
        convnet::ConvNet &m_net;
        convnet::NetworkState m_states[3];
        std::unique_ptr<convnet::ExecutionWorkspace> m_executionWorkspace;
        std::unique_ptr<convnet::ExecutionStream> m_executionStream;
};


#endif // LAYERTEST_H
