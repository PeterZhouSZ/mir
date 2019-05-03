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

#include "ReluLayer.h"

namespace convnet {

const char *ReluLayer::TypeStr = "relu";

ReluLayer::ReluLayer(ConvNet &convnet, std::string name) : Layer(convnet, std::move(name))
{
    m_inputNames.push_back("");
    m_outputNames.push_back("");
}

void ReluLayer::save(tinyxml2::XMLElement *rootElement) const
{
    tinyxml2::XMLElement *element = rootElement->GetDocument()->NewElement("layer");
    rootElement->LinkEndChild(element);

    element->SetAttribute("type", TypeStr);
    element->SetAttribute("name", m_name.c_str());

    saveInputOutputNames(element);
}

std::string ReluLayer::toString() const
{
    std::stringstream str;
    str << TypeStr << " on " << m_connectionList->outputs[m_inputs[0]].size;
    return str.str();
}

void ReluLayer::resize(ExecutionStream &/*stream*/, Key<ConvNet>)
{
    if (m_inputs.size() != 1)
        throw std::runtime_error("Relu inputs must be 1");

    if (m_outputs.size() != 1)
        throw std::runtime_error("Relu outputs must be 1");


    const TensorSize &inputSize = m_connectionList->outputs[m_inputs[0]].size;
    TensorSize &outputSize = m_connectionList->outputs[m_outputs[0]].size;

    outputSize = inputSize;
}

}

