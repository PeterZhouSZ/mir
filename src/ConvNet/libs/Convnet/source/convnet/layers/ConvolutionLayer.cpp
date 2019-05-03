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

#include "ConvolutionLayer.h"

#include <iostream>

namespace convnet {

void ConvolutionParameters::seedParametersGaussian(unsigned numFilters, const TensorSize &size, std::mt19937 &rng, float sigmaWeights, float sigmaBias)
{
    std::normal_distribution<float> dist(0.0f, sigmaWeights);
    std::normal_distribution<float> biasDist(0.0f, sigmaBias);

    m_filterSize = size;
    m_kernel.resize(numFilters);
    for (auto &k : m_kernel) {
        float filterNorm = 0.0f;
        k.filter.allocate(size);
        for (unsigned z = 0; z < size.depth; z++)
            for (unsigned y = 0; y < size.height; y++)
                for (unsigned x = 0; x < size.width; x++)
                    for (unsigned c = 0; c < size.channels; c++) {
                        k.filter(x, y, z, c, 0) = dist(rng);
                        filterNorm += k.filter(x, y, z, c, 0)*k.filter(x, y, z, c, 0);
                    }
        filterNorm = std::sqrt(filterNorm);
        k.bias = biasDist(rng) * filterNorm;
    }

    structureChanged();
}

void ConvolutionParameters::seedParametersGaussianXavier(unsigned numFilters, const TensorSize &size, std::mt19937 &rng, 
                                                         bool adaptLR, float learningRateScale,
                                                         bool orthonormal, unsigned duplicateChannels, bool duplicateSpacially)
{
    m_filterSize = size;

    //float varWeights = 1.0f / ((numFilters + size.channels)*0.5f);
    float varWeights = 1.0f / size.numElements();
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f*varWeights));
    std::cout << "Initializing with " << std::sqrt(2.0f*varWeights) << std::endl;
    std::normal_distribution<float> biasDist(0.0f, 1.0f);//std::sqrt(1.0f / numFilters));
    
    unsigned numElems = duplicateSpacially?m_filterSize.channels:m_filterSize.numElements();
    
    Eigen::MatrixXf weights;
    weights.resize(numFilters/duplicateChannels, numElems);
    
    for (unsigned k = 0; k < numFilters/duplicateChannels; k++) {
        for (unsigned i = 0; i < numElems; i++)
            weights(k, i) = dist(rng);
    }
    
    if (orthonormal) {
        if (numFilters/duplicateChannels > numElems) 
            throw std::runtime_error("Too many filters for them to be all orthogonal!");
        
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(weights, Eigen::ComputeThinU | Eigen::ComputeThinV);
        weights = svd.matrixV().transpose();
    }



    m_kernel.resize(numFilters);
    for (unsigned kernelIdx = 0; kernelIdx < m_kernel.size(); kernelIdx++) {
        auto &k = m_kernel[kernelIdx];
        k.filter.allocate(size);
        for (unsigned c = 0; c < size.channels; c++) {
            if (!duplicateSpacially) {
                for (unsigned z = 0; z < size.depth; z++)
                    for (unsigned y = 0; y < size.height; y++)
                        for (unsigned x = 0; x < size.width; x++) {
                            k.filter(x, y, z, c, 0) = weights(kernelIdx/duplicateChannels,
                                x + y * size.width + z * size.width * size.height + c * size.width * size.height * size.depth
                            );
                        }
            } else {
                for (unsigned z = 0; z < size.depth; z++)
                    for (unsigned y = 0; y < size.height; y++)
                        for (unsigned x = 0; x < size.width; x++)
                            k.filter(x, y, z, c, 0) = weights(kernelIdx/duplicateChannels, c);
            }
        }
        k.bias = biasDist(rng);
    }

    if (adaptLR)
        setParameterLearningXavier(learningRateScale);

    structureChanged();
}

void ConvolutionParameters::setParameterLearningXavier(float scale)
{
    m_biasLearningRate =
    m_weightLearningRate = scale / std::sqrt((float)m_filterSize.numElements());
}



float ConvolutionParameters::computeRegularizationLoss(ExecutionWorkspace &workspace, ExecutionStream &stream)
{
    pullFromBackend(workspace, stream);
    float sum = 0.0f;
    for (auto &k : m_kernel) {
        float kernelSum = 0.0f;
        
        for (unsigned z = 0; z < m_filterSize.depth; z++)
            for (unsigned y = 0; y < m_filterSize.height; y++)
                for (unsigned x = 0; x < m_filterSize.width; x++)
                    for (unsigned c = 0; c < m_filterSize.channels; c++) {
                        kernelSum += k.filter(x, y, z, c, 0)*k.filter(x, y, z, c, 0) * 0.5f * m_weightDecay;
                    }
        sum += kernelSum + k.bias*k.bias * 0.5f * m_weightDecay;
    }
    return sum;
}


const char *ConvolutionLayer::TypeStr = "conv";


ConvolutionLayer::ConvolutionLayer(ConvNet &convnet, std::string name) : Layer(convnet, std::move(name))
{
    m_inputNames.push_back("");
    m_outputNames.push_back("");

    m_padding.setZero();
    m_stride << 1,1,1;
    m_upsample << 1,1,1;
}

ConvolutionLayer *ConvolutionLayer::seedParametersGaussian(unsigned numFilters, const TensorSize &size, std::mt19937 &rng, float sigmaWeights, float sigmaBias)
{
    m_parameters->seedParametersGaussian(numFilters, size, rng, sigmaWeights, sigmaBias);
    return this;
}

ConvolutionLayer *ConvolutionLayer::seedParametersGaussianXavier(unsigned numFilters, const TensorSize &size, std::mt19937 &rng, 
                                                                 bool adaptLR, float learningRateScale,
                                                                 bool orthonormal, unsigned duplicateChannels, bool duplicateSpacially)
{
    m_parameters->seedParametersGaussianXavier(numFilters, size, rng, adaptLR, learningRateScale, orthonormal, duplicateChannels, duplicateSpacially);
    return this;
}

ConvolutionLayer *ConvolutionLayer::setParameterLambdas(float lambda1, float lambda2)
{
    m_parameters->m_lambda1 = lambda1;
    m_parameters->m_lambda2 = lambda2;
    return this;
}

void ConvolutionLayer::setupParameters(ConvNet &convnet, std::string parameterName, Key<ConvNet>)
{
    if (!parameterName.empty()) {
        LearnedParameters *params = convnet.getParameterSet().findByName(parameterName);
        if (params == nullptr) {
            m_parameters = instantiateParameters(std::move(parameterName));
            convnet.getParameterSet().learnedParameters.push_back(std::unique_ptr<LearnedParameters>(m_parameters));
        } else {
            m_parameters = dynamic_cast<ConvolutionParameters*>(params);
            if (m_parameters == nullptr)
                throw std::runtime_error(std::string("Incompatible parameters found for ConvolutionalLayer under ")+parameterName);
        }
    } else {
        m_parameters = instantiateParameters("");
        convnet.getParameterSet().learnedParameters.push_back(std::unique_ptr<LearnedParameters>(m_parameters));
    }
}

void ConvolutionLayer::loadParameters(ConvNet &convnet, tinyxml2::XMLElement *element, Key<ConvNet>)
{
    loadInputOutputNames(element);

    bool initialize = true;

    {
        tinyxml2::XMLElement *vec = element->FirstChildElement("padding");
        if (vec != nullptr) {
            if (vec->QueryAttribute("x", &m_padding[0]) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading padding!");
            if (vec->QueryAttribute("y", &m_padding[1]) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading padding!");
            if (vec->QueryAttribute("z", &m_padding[2]) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading padding!");
        }
    }
    {
        tinyxml2::XMLElement *vec = element->FirstChildElement("stride");
        if (vec != nullptr) {
            if (vec->QueryAttribute("x", &m_stride[0]) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading stride!");
            if (vec->QueryAttribute("y", &m_stride[1]) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading stride!");
            if (vec->QueryAttribute("z", &m_stride[2]) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading stride!");
        }
    }
    {
        tinyxml2::XMLElement *vec =element->FirstChildElement("upsample");
        if (vec != nullptr) {
            if (vec->QueryAttribute("x", &m_upsample[0]) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading upsampling!");
            if (vec->QueryAttribute("y", &m_upsample[1]) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading upsampling!");
            if (vec->QueryAttribute("z", &m_upsample[2]) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading upsampling!");
        }
    }

    const char *parameterName = element->Attribute("parameterName");
    if ((parameterName != nullptr) && (parameterName[0] != 0)) {
        LearnedParameters *params = convnet.getParameterSet().findByName(parameterName);
        if (params == nullptr) {
            m_parameters = instantiateParameters(parameterName);
            convnet.getParameterSet().learnedParameters.push_back(std::unique_ptr<LearnedParameters>(m_parameters));
        } else {
            m_parameters = dynamic_cast<ConvolutionParameters*>(params);
            if (m_parameters == nullptr)
                throw std::runtime_error(std::string("Incompatible parameters found under ")+parameterName);
            initialize = false;
        }
    } else {
        m_parameters = instantiateParameters("");
        convnet.getParameterSet().learnedParameters.push_back(std::unique_ptr<LearnedParameters>(m_parameters));
    }

    if (initialize) {
        if (element->QueryAttribute("width", &m_parameters->m_filterSize.width) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading filter size width!");
        if (element->QueryAttribute("height", &m_parameters->m_filterSize.height) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading filter size height!");
        if (element->QueryAttribute("depth", &m_parameters->m_filterSize.depth) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading filter size depth!");
        if (element->QueryAttribute("channels", &m_parameters->m_filterSize.channels) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading filter size channels!");
        if (element->QueryAttribute("weight_decay", &m_parameters->m_weightDecay) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading filter weight decay!");
        
        {
            const char *bias = element->Attribute("has_bias");
            if (bias != nullptr) {
                if (std::string("true") == bias)
                    m_parameters->m_hasBias = true;
                else if (std::string("false") == bias)
                    m_parameters->m_hasBias = false;
                else
                    throw std::runtime_error("Invalid value for 'has_bias'!");
            } else {
                m_parameters->m_hasBias = true;
            }
        }

        {
            tinyxml2::XMLElement *vec =element->FirstChildElement("clip");
            if (vec != nullptr) {
                if (vec->QueryAttribute("clip_min", &m_parameters->m_clipMin) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading clipping!");
                if (vec->QueryAttribute("clip_max", &m_parameters->m_clipMax) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading clipping!");
            }
        }
        
        m_parameters->m_normalization = ConvolutionParameters::NORM_NONE;
        {
            tinyxml2::XMLElement *norm = element->FirstChildElement("normalization");
            if (norm != nullptr) {
                const char *type = norm->Attribute("type");
                if (type == nullptr) throw std::runtime_error("Type missing in normalization tag!");
                if (std::string("none") == type) {
                } else
                if (std::string("running_avg_bn") == type) {
                    if (norm->QueryAttribute("lambda", &m_parameters->m_runningAvgLambda) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error normalization lambda!");
                } else
                    throw std::runtime_error("Unknown normalization type!");
            }
        }


        for (tinyxml2::XMLElement *kernelElement = element->FirstChildElement("kernel"); kernelElement != nullptr; kernelElement = kernelElement->NextSiblingElement("kernel")) {
            ConvolutionParameters::Kernel kernel;
            if (m_parameters->m_hasBias) {
                if (kernelElement->QueryAttribute("bias", &kernel.bias) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading filter bias!");
            } else {
                kernel.bias = 0.0f;
            }

            const char *coeffStr = kernelElement->GetText();
            if (coeffStr == nullptr) throw std::runtime_error("Coeffs not found!");

            std::stringstream coeffs(coeffStr);
            coeffs.exceptions(std::stringstream::failbit);

            kernel.filter.allocate(m_parameters->m_filterSize);
            for (unsigned z = 0; z < m_parameters->m_filterSize.depth; z++)
                for (unsigned y = 0; y < m_parameters->m_filterSize.height; y++)
                    for (unsigned x = 0; x < m_parameters->m_filterSize.width; x++)
                        for (unsigned c = 0; c < m_parameters->m_filterSize.channels; c++)
                            try {
                                coeffs >> kernel.filter(x, y, z, c, 0);
                            } catch (const std::exception &e) {
                                std::cout << coeffStr << std::endl;
                                std::cout << "kernel " << m_parameters->m_kernel.size() << std::endl;
                                std::cout << x << " " << y << " " << z << " " << c << std::endl;
                                if (coeffs.rdstate() & std::ios_base::badbit)
                                    std::cout << "std::ios_base::badbit" << std::endl;

                                if (coeffs.rdstate() & std::ios_base::failbit)
                                    std::cout << "std::ios_base::failbit" << std::endl;

                                if (coeffs.rdstate() & std::ios_base::eofbit)
                                    std::cout << "std::ios_base::eofbit" << std::endl;

                                if (coeffs.rdstate() & std::ios_base::goodbit)
                                    std::cout << "std::ios_base::goodbit" << std::endl;
  
                                throw std::runtime_error(std::string("Error loading convolution kernel: ")+e.what());                
                            }

            m_parameters->m_kernel.push_back(kernel);
        }

        #if 1
            m_parameters->m_biasLearningRate =
            m_parameters->m_weightLearningRate = 1.0f / std::sqrt((float)m_parameters->m_filterSize.numElements());
        #endif

        m_parameters->structureChanged();
    }
    std::cout << "Read convolution layer with " << m_parameters->m_kernel.size() << " filters of: " << m_parameters->m_filterSize << std::endl;
}



void ConvolutionLayer::save(tinyxml2::XMLElement *rootElement) const
{
    tinyxml2::XMLElement *element = rootElement->GetDocument()->NewElement("layer");
    rootElement->LinkEndChild(element);

    element->SetAttribute("type", TypeStr);
    element->SetAttribute("name", m_name.c_str());

    {
        tinyxml2::XMLElement *vec = rootElement->GetDocument()->NewElement("padding");
        element->LinkEndChild(vec);
        vec->SetAttribute("x", m_padding[0]);
        vec->SetAttribute("y", m_padding[1]);
        vec->SetAttribute("z", m_padding[2]);
    }
    {
        tinyxml2::XMLElement *vec = rootElement->GetDocument()->NewElement("stride");
        element->LinkEndChild(vec);
        vec->SetAttribute("x", m_stride[0]);
        vec->SetAttribute("y", m_stride[1]);
        vec->SetAttribute("z", m_stride[2]);
    }
    {
        tinyxml2::XMLElement *vec = rootElement->GetDocument()->NewElement("upsample");
        element->LinkEndChild(vec);
        vec->SetAttribute("x", m_upsample[0]);
        vec->SetAttribute("y", m_upsample[1]);
        vec->SetAttribute("z", m_upsample[2]);
    }

    element->SetAttribute("parameterName", m_parameters->getName().c_str());

    element->SetAttribute("width", m_parameters->m_filterSize.width);
    element->SetAttribute("height", m_parameters->m_filterSize.height);
    element->SetAttribute("depth", m_parameters->m_filterSize.depth);
    element->SetAttribute("channels", m_parameters->m_filterSize.channels);

    element->SetAttribute("weight_decay", m_parameters->m_weightDecay);
    
    if (m_parameters->m_hasBias) 
        element->SetAttribute("has_bias", "true");
    else
        element->SetAttribute("has_bias", "false");
    
        
    
    switch (m_parameters->m_normalization) {
        case ConvolutionParameters::NORM_NONE: {
            tinyxml2::XMLElement *norm = rootElement->GetDocument()->NewElement("normalization");
            element->LinkEndChild(norm);
            norm->SetAttribute("type", "none");
        } break;
        case ConvolutionParameters::NORM_RUNNING_AVG_BN: {
            tinyxml2::XMLElement *norm = rootElement->GetDocument()->NewElement("normalization");
            element->LinkEndChild(norm);
            norm->SetAttribute("type", "running_avg_bn");
            norm->SetAttribute("lambda", m_parameters->m_runningAvgLambda);
        } break;
    }

    {
        tinyxml2::XMLElement *vec = rootElement->GetDocument()->NewElement("clip");
        element->LinkEndChild(vec);
        vec->SetAttribute("clip_min", m_parameters->m_clipMin);
        vec->SetAttribute("clip_max", m_parameters->m_clipMax);
    }

    for (const auto &kernel : m_parameters->m_kernel) {
        tinyxml2::XMLElement *kernelElement = rootElement->GetDocument()->NewElement("kernel");
        element->LinkEndChild(kernelElement);
        kernelElement->SetAttribute("bias", kernel.bias);
        std::stringstream allCoeffs;
        for (unsigned z = 0; z < m_parameters->m_filterSize.depth; z++)
            for (unsigned y = 0; y < m_parameters->m_filterSize.height; y++)
                for (unsigned x = 0; x < m_parameters->m_filterSize.width; x++)
                    for (unsigned c = 0; c < m_parameters->m_filterSize.channels; c++)
                        allCoeffs << " " << kernel.filter(x, y, z, c, 0);
        kernelElement->SetText(allCoeffs.str().c_str());
    }

    saveInputOutputNames(element);
}

void ConvolutionLayer::resize(ExecutionStream &/*stream*/, Key<ConvNet>)
{
    if (m_inputs.size() != 1)
        throw std::runtime_error("Convolution inputs must be 1");

    if (m_outputs.size() != 1)
        throw std::runtime_error("Convolution outputs must be 1");


    const TensorSize &inputSize = m_connectionList->outputs[m_inputs[0]].size;
    TensorSize &outputSize = m_connectionList->outputs[m_outputs[0]].size;


    if (inputSize.width * m_upsample[0] + 2*m_padding[0] <= (m_parameters->m_filterSize.width-1) * m_stride[0])
        throw std::runtime_error("Input width too small for filter size!");
    if (inputSize.height * m_upsample[1] + 2*m_padding[1] <= (m_parameters->m_filterSize.height-1) * m_stride[1])
        throw std::runtime_error("Input height too small for filter size!");
    if (inputSize.depth * m_upsample[2] + 2*m_padding[2] <= (m_parameters->m_filterSize.depth-1) * m_stride[2])
        throw std::runtime_error("Input depth too small for filter size!");

    if (inputSize.channels != m_parameters->m_filterSize.channels) {
        std::cout << "Input size: " << inputSize << std::endl;
        std::cout << "filter channels: " << m_parameters->m_filterSize.channels << std::endl;
        
        throw std::runtime_error(std::string("Channels don't match! Layer:") + m_name);
    }
    outputSize = inputSize;

    outputSize.width =  (inputSize.width    * m_upsample[0] + 2*m_padding[0] - m_parameters->m_filterSize.width)     / m_stride[0] + 1;
    outputSize.height = (inputSize.height   * m_upsample[1] + 2*m_padding[1] - m_parameters->m_filterSize.height)    / m_stride[1] + 1;
    outputSize.depth =  (inputSize.depth    * m_upsample[2] + 2*m_padding[2] - m_parameters->m_filterSize.depth)     / m_stride[2] + 1;
    outputSize.channels = m_parameters->m_kernel.size();
}

std::string ConvolutionLayer::toString() const
{
    std::stringstream str;
    str << "Conv " << m_parameters->m_filterSize << " on " << m_connectionList->outputs[m_inputs[0]].size;
    return str.str();
}


}
