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

#ifndef BATCHNORMLAYER_H
#define BATCHNORMLAYER_H

#include "Layer.h"

#include <Eigen/Dense>

#include <random>

namespace convnet {

class BatchNormParameters : public LearnedParameters
{
    public:
        struct Channel {
            float runningMean = 0.0f;
            float runningVar = 1.0f;
            
            float scale = 1.0f;
            float bias = 0.0f;
        };
        
        float m_scaleLearningRate = 1.0f;
        float m_biasLearningRate = 1.0f;
        
        float m_lambda1 = 0.995f;
        float m_lambda2 = 0.95f;
        
        float m_momentLambda = 0.9f;
        
        float epsilon = 1e-4f;
        
        bool m_adaptScale = false;
        bool m_adaptBias = true;

        BatchNormParameters(std::string name) : LearnedParameters(std::move(name)) { }

        void initialize(unsigned numChannels, std::mt19937 &rng);
        
        virtual void structureChanged() { }
        
        std::vector<Channel> m_channels;
};

class BatchNormLayer : public Layer
{
    public:
        static const char *TypeStr;

        BatchNormLayer(ConvNet &convnet, std::string name = std::string());

        virtual void save(tinyxml2::XMLElement *rootElement) const override;

        virtual std::string toString() const override;

        BatchNormParameters *getParameters() { return m_parameters; }
        const BatchNormParameters *getParameters() const { return m_parameters; }

        BatchNormLayer *initialize(unsigned numChannels, std::mt19937 &rng) { m_parameters->initialize(numChannels, rng); return this; }
        BatchNormLayer *setParameterLambdas(float lambda1, float lambda2) { m_parameters->m_lambda1 = lambda1; m_parameters->m_lambda2 = lambda2; return this; }
        BatchNormLayer *setMomentLambda(float lambda) { m_parameters->m_momentLambda = lambda; return this; }
        BatchNormLayer *setLearningRateFactor(float factor) { m_parameters->m_scaleLearningRate = m_parameters->m_biasLearningRate = factor; return this; }
//        inline BatchNormLayer *setBiasDecay(float biasDecay) { m_parameters->m_biasDecay = biasDecay; return this; }
        
        virtual void setupParameters(ConvNet &convnet, std::string parameterName, Key<ConvNet>) override;
        virtual void loadParameters(ConvNet &convnet, tinyxml2::XMLElement *element, Key<ConvNet>) override;   

        virtual void resize(ExecutionStream &stream, Key<ConvNet>) override;
    protected:
        virtual BatchNormParameters *instantiateParameters(std::string name) = 0;

        BatchNormParameters *m_parameters;
        
        bool m_backpropAlwaysTrainigMode = true;
};

}


#endif // BATCHNORMLAYER_H
