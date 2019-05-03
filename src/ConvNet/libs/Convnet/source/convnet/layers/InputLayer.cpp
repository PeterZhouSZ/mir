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

#include "InputLayer.h"
#include <iostream>

namespace convnet {

const char *InputLayer::TypeStr = "input";

InputLayer::InputLayer(ConvNet &convnet, std::string name) : Layer(convnet, std::move(name))
{
}

void InputLayer::swapInStagingBuffer(NetworkState& state, InputStagingBuffer& stagingBuffer) const
{
    for (unsigned i = 0; i < m_outputs.size(); i++)
        std::swap(state.outputs[m_outputs[i]], stagingBuffer.outputs[i]);
}


InputLayer* InputLayer::setInputSizes(std::vector<TensorSize> inputSizes)
{
    m_inputSizes = std::move(inputSizes);
    return this;
}

void InputLayer::save(tinyxml2::XMLElement *rootElement) const
{
    tinyxml2::XMLElement *element = rootElement->GetDocument()->NewElement("layer");
    rootElement->LinkEndChild(element);

    element->SetAttribute("type", TypeStr);
    element->SetAttribute("name", m_name.c_str());

    saveInputOutputNames(element);
}

std::string InputLayer::toString() const
{
    return TypeStr;
}

void InputLayer::resize(ExecutionStream &/*stream*/, Key<ConvNet>)
{
    if (m_inputs.size() != 0)
        throw std::runtime_error("InputLayer inputs must be 0");

    if (m_outputs.size() != m_inputSizes.size())
        throw std::runtime_error("InputLayer outputs must match specified sizes");


    for (unsigned i = 0; i < m_inputSizes.size(); i++)
        m_connectionList->outputs[m_outputs[i]].size = m_inputSizes[i];
}

}
