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

#include "CrossEntropyLossLayer.h"


namespace convnet {

const char *CrossEntropyLossLayer::TypeStr = "cross_entropy_loss";

CrossEntropyLossLayer::CrossEntropyLossLayer(ConvNet &convnet, std::string name) : Layer(convnet, std::move(name))
{
    m_outputNames.push_back("");
}

void CrossEntropyLossLayer::save(tinyxml2::XMLElement *rootElement) const
{
    tinyxml2::XMLElement *element = rootElement->GetDocument()->NewElement("layer");
    rootElement->LinkEndChild(element);

    element->SetAttribute("type", TypeStr);
    element->SetAttribute("name", m_name.c_str());
    element->SetAttribute("weight", m_weight);

    saveInputOutputNames(element);
}

void CrossEntropyLossLayer::loadParameters(ConvNet &/*convnet*/, tinyxml2::XMLElement *element, Key<ConvNet>)
{
    loadInputOutputNames(element);

    if (element->QueryAttribute("weight", &m_weight) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading CrossEntropyLossLayer weight!");
}



std::string CrossEntropyLossLayer::toString() const
{
    std::stringstream str;
    str << TypeStr << " on " << m_connectionList->outputs[m_inputs[0]].size;
    return str.str();
}

void CrossEntropyLossLayer::resize(ExecutionStream &/*stream*/, Key<ConvNet>)
{
    if (m_inputs.size() != 2)
        throw std::runtime_error("CrossEntropyLossLayer inputs must be 2");

    if (m_outputs.size() != 1)
        throw std::runtime_error("CrossEntropyLossLayer outputs must be 1");


    const TensorSize &inputSize0 = m_connectionList->outputs[m_inputs[0]].size;
    const TensorSize &inputSize1 = m_connectionList->outputs[m_inputs[1]].size;
    TensorSize &outputSize = m_connectionList->outputs[m_outputs[0]].size;

    outputSize.width = std::min(inputSize0.width, inputSize1.width);
    outputSize.height = std::min(inputSize0.height, inputSize1.height);
    outputSize.depth = std::min(inputSize0.depth, inputSize1.depth);
    outputSize.channels = 1;
}

}

