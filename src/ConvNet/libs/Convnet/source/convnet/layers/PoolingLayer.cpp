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

#include "PoolingLayer.h"

#include <iostream>

namespace convnet {

const char *PoolingLayer::TypeStr = "pool";

PoolingLayer::PoolingLayer(ConvNet &convnet, std::string name) : Layer(convnet, std::move(name))
{
    m_windowSize[0] = 1;
    m_windowSize[1] = 1;
    m_windowSize[2] = 1;

    m_inputNames.push_back("");
    m_outputNames.push_back("");
}

void PoolingLayer::save(tinyxml2::XMLElement *rootElement) const
{
    tinyxml2::XMLElement *element = rootElement->GetDocument()->NewElement("layer");
    rootElement->LinkEndChild(element);

    element->SetAttribute("type", TypeStr);
    element->SetAttribute("name", m_name.c_str());
    switch (m_mode) {
        case MODE_AVG:
            element->SetAttribute("mode", "avg");
        break;
        case MODE_MAX:
            element->SetAttribute("mode", "max");
        break;
    }

    saveInputOutputNames(element);

    tinyxml2::XMLElement *windowElement = rootElement->GetDocument()->NewElement("window");
    element->LinkEndChild(windowElement);

    windowElement->SetAttribute("width", m_windowSize[0]);
    windowElement->SetAttribute("height", m_windowSize[1]);
    windowElement->SetAttribute("depth", m_windowSize[2]);
}

std::string PoolingLayer::toString() const
{
    std::stringstream str;
    str << "Pool " << m_windowSize[0]<<"x"<<m_windowSize[1]<<"x"<<m_windowSize[2] << " on " << m_connectionList->outputs[m_inputs[0]].size;
    return str.str();
}

PoolingLayer *PoolingLayer::setWindowSize(unsigned w, unsigned h, unsigned d)
{
    m_windowSize[0] = w;
    m_windowSize[1] = h;
    m_windowSize[2] = d;

    structureChanged();
    return this;
}

void PoolingLayer::loadParameters(ConvNet &/*convnet*/, tinyxml2::XMLElement *element, Key<ConvNet>)
{
    loadInputOutputNames(element);

    tinyxml2::XMLElement *windowElement = element->FirstChildElement("window");
    if (windowElement == nullptr)
        throw std::runtime_error("No window element found!");

    if (windowElement->QueryAttribute("width", &m_windowSize[0]) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading window width!");
    if (windowElement->QueryAttribute("height", &m_windowSize[1]) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading window height!");
    if (windowElement->QueryAttribute("depth", &m_windowSize[2]) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading window depth!");

    const char *modeStr = element->Attribute("mode");
    if (modeStr == nullptr)
        throw std::runtime_error("No mode attribute found!");

    if (std::string(modeStr) == "avg")
        m_mode = MODE_AVG; else
    if (std::string(modeStr) == "max")
        m_mode = MODE_MAX;
    else
        throw std::runtime_error("Invalid mode attribute found!");


    structureChanged();
    std::cout << "Read pooling layer of size " << m_windowSize[0]<<"x"<<m_windowSize[1]<<"x"<<m_windowSize[2] << std::endl;
}

void PoolingLayer::resize(ExecutionStream &/*stream*/, Key<ConvNet>)
{
    if (m_inputs.size() != 1)
        throw std::runtime_error("PoolingLayer inputs must be 1");

    if (m_outputs.size() != 1)
        throw std::runtime_error("PoolingLayer outputs must be 1");


    const TensorSize &inputSize = m_connectionList->outputs[m_inputs[0]].size;
    TensorSize &outputSize = m_connectionList->outputs[m_outputs[0]].size;

    outputSize = inputSize;

    if (inputSize.width % m_windowSize[0] != 0)
        std::cout << "Warning: input width not divisible by pool layer width!" << std::endl;

    if (inputSize.height % m_windowSize[1] != 0)
        std::cout << "Warning: input height not divisible by pool layer height!" << std::endl;

    if (inputSize.depth % m_windowSize[2] != 0)
        std::cout << "Warning: input depth not divisible by pool layer depth!" << std::endl;


    outputSize.width = inputSize.width / m_windowSize[0];
    outputSize.height = inputSize.height / m_windowSize[1];
    outputSize.depth = inputSize.depth / m_windowSize[2];
}


}
