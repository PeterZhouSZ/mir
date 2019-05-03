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

#ifndef TRAINER_H
#define TRAINER_H


#include "../NetworkState.h"
#include "../ExecutionWorkspace.h"
#include "../ExecutionStream.h"

#include <tools/TaskScheduler.h>
#include <thread>
#include <mutex>
#include <utility>

#include <random>

namespace convnet {

class InputLayer;
class ConvNet;
class TrainingDataSource;

class Trainer
{
    public:
        Trainer(ConvNet &convnet, TrainingDataSource &dataSource);

        virtual bool learn(unsigned numBatchesPerStep, float stepsize);
        virtual void run();
        
        inline NetworkState &getState() { return m_state; }
        inline ExecutionWorkspace &getExecutionWorkspace() { return *m_executionWorkspace; }
        inline ExecutionStream &getComputeStream() { return *m_computeStream; }
        
        void runDataset(TrainingDataSource &validationSet, std::function<bool(unsigned, NetworkState&)> dataCallback);
    protected:
        virtual void handleLearningLoss() { }
        
        float sumLoss(unsigned output);
        
        ConvNet &m_convnet;
        TrainingDataSource &m_dataSource;

        InputLayer *m_inputLayer;
        NetworkState m_state;
        std::unique_ptr<ExecutionWorkspace> m_executionWorkspace;
        std::unique_ptr<ExecutionStream> m_computeStream;
        std::unique_ptr<ExecutionStreamWaitingFence> m_forwardDoneFence;
        std::unique_ptr<ExecutionStreamWaitingFence> m_backwardDoneFence;
        InputStagingBuffer m_stagingBuffer;
};

}

#endif // TRAINER_H
