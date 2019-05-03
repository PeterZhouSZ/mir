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

#ifndef CONVOLUTIONLAYER_H
#define CONVOLUTIONLAYER_H

#include "Layer.h"

#include "../Blobb.h"

#include <random>
#include <Eigen/Dense>

namespace convnet {

class ConvolutionParameters : public LearnedParameters
{
    public:
        struct Kernel {
            Blobb<1> filter;
            float bias;
        };

        TensorSize m_filterSize;
        std::vector<Kernel> m_kernel;
        float m_weightDecay = 0.01f;

        float m_lambda1 = 0.95f;
        float m_lambda2 = 0.95f;

        float m_weightLearningRate = 1.0f;
        float m_biasLearningRate = 1.0f;
    
        float m_clipMin = -1e30f;
        float m_clipMax = 1e30f;
        
        bool m_hasBias = true;
        
        enum InPlaceNormalization {
            NORM_NONE,
            NORM_RUNNING_AVG_BN
        };
        InPlaceNormalization m_normalization = NORM_NONE;
        float m_runningAvgLambda = 0.995f;

        ConvolutionParameters(std::string name) : LearnedParameters(std::move(name)) { }

        void seedParametersGaussian(unsigned numFilters, const TensorSize &size, std::mt19937 &rng, float sigmaWeights = 0.001f, float sigmaBias = 1.001f);
        void seedParametersGaussianXavier(unsigned numFilters, const TensorSize &size, std::mt19937 &rng, 
                                          bool adaptLR = true, float learningRateScale = 1.0f,
                                          bool orthonormal = false, unsigned duplicateChannels = 1, bool duplicateSpacially = false
                                         );
        void setParameterLearningXavier(float scale);

        virtual float computeRegularizationLoss(ExecutionWorkspace &workspace, ExecutionStream &stream);
        
        virtual void structureChanged() { }
};

class ConvolutionLayer : public Layer
{
    public:
        static const char *TypeStr;

        ConvolutionLayer(ConvNet &convnet, std::string name = std::string());

        virtual void save(tinyxml2::XMLElement *rootElement) const override;

        virtual std::string toString() const override;

        ConvolutionParameters *getParameters() { return m_parameters; }
        const ConvolutionParameters *getParameters() const { return m_parameters; }

        ConvolutionLayer *seedParametersGaussian(unsigned numFilters, const TensorSize &size, std::mt19937 &rng, float sigmaWeights = 0.001f, float sigmaBias = 1.001f);
        ConvolutionLayer *seedParametersGaussianXavier(unsigned numFilters, const TensorSize &size, std::mt19937 &rng, 
                                                       bool adaptLR = true, float learningRateScale = 1.0f,
                                                        bool orthonormal = false, unsigned duplicateChannels = 1, bool duplicateSpacially = false);
        ConvolutionLayer *setParameterLambdas(float lambda1, float lambda2);
        ConvolutionLayer *setParameterClip(float clipMin, float clipMax) { m_parameters->m_clipMin = clipMin; m_parameters->m_clipMax = clipMax; return this; }
        ConvolutionLayer *setParameterLearningRate(float weight, float bias) { m_parameters->m_weightLearningRate = weight; m_parameters->m_biasLearningRate = bias; return this; }
        ConvolutionLayer *setParameterLearningXavier(float scale) { m_parameters->setParameterLearningXavier(scale); return this; }
        ConvolutionLayer *setParameterNormalizationRunningAvgBN(float lambda = 0.995f) { m_parameters->m_normalization = ConvolutionParameters::NORM_RUNNING_AVG_BN; m_parameters->m_runningAvgLambda = lambda; return this; }
        ConvolutionLayer *setHasBias(bool hasBias) { m_parameters->m_hasBias = hasBias; return this; }
        inline ConvolutionLayer *setWeightDecay(float weightDecay) { m_parameters->m_weightDecay = weightDecay; return this; }
        
        virtual void setupParameters(ConvNet &convnet, std::string parameterName, Key<ConvNet>) override;
        virtual void loadParameters(ConvNet &convnet, tinyxml2::XMLElement *element, Key<ConvNet>) override;   

        virtual void resize(ExecutionStream &stream, Key<ConvNet>) override;
        
        ConvolutionLayer *setPadding(const Eigen::Vector3i &padding) { m_padding = padding; return this; }
        ConvolutionLayer *setStride(const Eigen::Vector3i &stride) { m_stride = stride; return this; }
        ConvolutionLayer *setUpsample(const Eigen::Vector3i &upsample) { m_upsample = upsample; return this; }
    protected:
        virtual ConvolutionParameters *instantiateParameters(std::string name) = 0;

        ConvolutionParameters *m_parameters;

        Eigen::Vector3i m_padding;
        Eigen::Vector3i m_stride;
        Eigen::Vector3i m_upsample;

};

}

#endif // CONVOLUTIONLAYER_H
