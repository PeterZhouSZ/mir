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

#include "BatchNormLayer.h"


#include <iostream>

namespace convnet {

void BatchNormParameters::initialize(unsigned numChannels, std::mt19937 &rng)
{
    std::normal_distribution<float> biasDist(0.0f, 1.0f);

    m_channels.resize(numChannels);
    for (auto &c : m_channels) {
        c.runningMean = 0.0f;
        c.runningVar = 1.0f;
        
        c.scale = 1.0f;
        c.bias = biasDist(rng);
    }

    structureChanged();
}


const char *BatchNormLayer::TypeStr = "batch_norm";


BatchNormLayer::BatchNormLayer(ConvNet &convnet, std::string name) : Layer(convnet, std::move(name))
{
    m_inputNames.push_back("");
    m_outputNames.push_back("");
}


void BatchNormLayer::setupParameters(ConvNet &convnet, std::string parameterName, Key<ConvNet>)
{
    if (!parameterName.empty()) {
        LearnedParameters *params = convnet.getParameterSet().findByName(parameterName);
        if (params == nullptr) {
            m_parameters = instantiateParameters(std::move(parameterName));
            convnet.getParameterSet().learnedParameters.push_back(std::unique_ptr<LearnedParameters>(m_parameters));
        } else {
            m_parameters = dynamic_cast<BatchNormParameters*>(params);
            if (m_parameters == nullptr)
                throw std::runtime_error(std::string("Incompatible parameters found for BatchNormLayer under ")+parameterName);
        }
    } else {
        m_parameters = instantiateParameters("");
        convnet.getParameterSet().learnedParameters.push_back(std::unique_ptr<LearnedParameters>(m_parameters));
    }
}

void BatchNormLayer::loadParameters(ConvNet &convnet, tinyxml2::XMLElement *element, Key<ConvNet>)
{
    loadInputOutputNames(element);

    bool initialize = true;

    const char *parameterName = element->Attribute("parameterName");
    if ((parameterName != nullptr) && (parameterName[0] != 0)) {
        LearnedParameters *params = convnet.getParameterSet().findByName(parameterName);
        if (params == nullptr) {
            m_parameters = instantiateParameters(parameterName);
            convnet.getParameterSet().learnedParameters.push_back(std::unique_ptr<LearnedParameters>(m_parameters));
        } else {
            m_parameters = dynamic_cast<BatchNormParameters*>(params);
            if (m_parameters == nullptr)
                throw std::runtime_error(std::string("Incompatible parameters found under ")+parameterName);
            initialize = false;
        }
    } else {
        m_parameters = instantiateParameters("");
        convnet.getParameterSet().learnedParameters.push_back(std::unique_ptr<LearnedParameters>(m_parameters));
    }

    if (initialize) {
        if (element->QueryAttribute("scale_lr", &m_parameters->m_scaleLearningRate) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading scale lr!");
        if (element->QueryAttribute("bias_lr", &m_parameters->m_biasLearningRate) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading bias lr!");
        if (element->QueryAttribute("lambda1", &m_parameters->m_lambda1) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading lambda1!");
        if (element->QueryAttribute("lambda2", &m_parameters->m_lambda2) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading lambda2!");
        if (element->QueryAttribute("moment_lambda", &m_parameters->m_momentLambda) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading moment lambda!");

        for (tinyxml2::XMLElement *channelElement = element->FirstChildElement("channel"); channelElement != nullptr; channelElement = channelElement->NextSiblingElement("channel")) {
            BatchNormParameters::Channel channel;
            if (channelElement->QueryAttribute("mean", &channel.runningMean) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading channel first moment!");
            if (channelElement->QueryAttribute("var", &channel.runningVar) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading channel second moment!");
            if (channelElement->QueryAttribute("scale", &channel.scale) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading channel scale!");
            if (channelElement->QueryAttribute("bias", &channel.bias) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading channel bias!");

            m_parameters->m_channels.push_back(channel);
        }

        m_parameters->structureChanged();
    }
    std::cout << "Read bn layer with " << m_parameters->m_channels.size() << " channels" << std::endl;
}



void BatchNormLayer::save(tinyxml2::XMLElement *rootElement) const
{
    tinyxml2::XMLElement *element = rootElement->GetDocument()->NewElement("layer");
    rootElement->LinkEndChild(element);

    element->SetAttribute("type", TypeStr);
    element->SetAttribute("name", m_name.c_str());

    element->SetAttribute("parameterName", m_parameters->getName().c_str());

    element->SetAttribute("scale_lr", m_parameters->m_scaleLearningRate);
    element->SetAttribute("bias_lr", m_parameters->m_biasLearningRate);
    element->SetAttribute("lambda1", m_parameters->m_lambda1);
    element->SetAttribute("lambda2", m_parameters->m_lambda2);
    element->SetAttribute("moment_lambda", m_parameters->m_momentLambda);

    for (const auto &channel : m_parameters->m_channels) {
        tinyxml2::XMLElement *channelElement = rootElement->GetDocument()->NewElement("channel");
        element->LinkEndChild(channelElement);
        channelElement->SetAttribute("mean", channel.runningMean);
        channelElement->SetAttribute("var", channel.runningVar);
        channelElement->SetAttribute("scale", channel.scale);
        channelElement->SetAttribute("bias", channel.bias);
    }

    saveInputOutputNames(element);
}

void BatchNormLayer::resize(ExecutionStream &/*stream*/, Key<ConvNet>)
{
    if (m_inputs.size() != 1)
        throw std::runtime_error("BatchNorm inputs must be 1");

    if (m_outputs.size() != 1)
        throw std::runtime_error("BatchNorm outputs must be 1");


    const TensorSize &inputSize = m_connectionList->outputs[m_inputs[0]].size;
    TensorSize &outputSize = m_connectionList->outputs[m_outputs[0]].size;
    outputSize = inputSize;
    
    if (inputSize.channels != m_parameters->m_channels.size()) {
        std::cout << "Input size: " << inputSize << std::endl;
        std::cout << "BN channels: " << m_parameters->m_channels.size() << std::endl;
        
        throw std::runtime_error(std::string("Channels don't match! Layer:") + m_name);
    }
}

std::string BatchNormLayer::toString() const
{
    std::stringstream str;
    str << "BatchNorm on " << m_connectionList->outputs[m_inputs[0]].size;
    return str.str();
}


}

