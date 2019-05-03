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

#ifndef CLASSIFICATIONTRAINER_H
#define CLASSIFICATIONTRAINER_H

#include "../NetworkState.h"
#include "../ExecutionWorkspace.h"
#include "../ExecutionStream.h"

#include "VAETrainer.h"

#include <tools/TaskScheduler.h>
#include <thread>
#include <mutex>
#include <utility>

#include <random>

#include <Eigen/Dense>

namespace convnet {

class HTTPReport;

class ClassificationTrainer
{
    public:
        ClassificationTrainer(ConvNet &convnet,
                   const std::vector<std::string> &rngSources,
                   TrainingDataSource &dataSource,
                   std::mt19937 &rng);

        bool learn(unsigned numBatchesPerStep, float stepsize);

        inline NetworkState &getState() { return m_state; }
        inline ExecutionWorkspace &getExecutionWorkspace() { return *m_executionWorkspace; }
        inline ExecutionStream &getComputeStream() { return *m_computeStream; }

        void debugOutputTest(TrainingDataSource &validationSet, const std::string &prefix, unsigned N);

        void runValidation(TrainingDataSource &validationSet, float &loss, float &accuracy, Eigen::MatrixXf &confusionMatrix, std::vector<unsigned> &totalPerClass, unsigned border = 0);
        void report(TrainingDataSource &validationSet, HTTPReport &report, const std::string &prefix,
            std::vector<std::vector<unsigned>> inputChannelSelection = std::vector<std::vector<unsigned>>(),
            std::vector<unsigned> outputChannelSelection = std::vector<unsigned>());

        inline float getLastBatchLoss() const { return m_lastLoss; }
        inline float getLastBatchAccuracy() const { return m_lastAccuracy; }
    protected:
        ConvNet &m_convnet;
        TrainingDataSource &m_dataSource;
        RngSource m_rngSource;

        float m_lastAccuracy = 0.0f;
        float m_lastLoss = 0.0f;

        unsigned m_outputLoss = ~0u;
        unsigned m_outputLabel = ~0u;
        unsigned m_outputRadarData = ~0u;
        unsigned m_outputInferredLabel = ~0u;

        const std::string m_fixedPrefix = "G_";

        InputLayer *m_inputLayer;
        NetworkState m_state;
        std::unique_ptr<ExecutionWorkspace> m_executionWorkspace;
        std::unique_ptr<ExecutionStream> m_computeStream;
        std::unique_ptr<ExecutionStreamWaitingFence> m_forwardDoneFence;
        std::unique_ptr<ExecutionStreamWaitingFence> m_backwardDoneFence;
        InputStagingBuffer m_stagingBuffer;
};

}



#endif // CLASSIFICATIONTRAINER_H
