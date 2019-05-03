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

#include "Layer.h"

#include "../Convnet.h"
#include "../FileBlob.h"

namespace convnet {

LearnedParameters::LearnedParameters(std::string name) : m_name(std::move(name))
{
}



Layer::Layer(ConvNet &convnet, std::string name) : m_convnet(convnet), m_name(std::move(name))
{
}


void Layer::connect(ConvNet::ConnectionList &connectionList, Key<ConvNet>)
{
    m_outputs.clear();
    m_inputs.clear();
    m_connectionList = &connectionList;

    if ((m_inputNames.size() == 1) && (m_inputNames[0].empty())) {
        if (connectionList.outputs.empty())
            throw std::runtime_error("Can't auto-connect to last output if first layer! Add an input layer first.");

        connectionList.outputs.back().consumer.push_back(this);
        m_inputs.push_back(connectionList.outputs.size()-1);
    } else {
        for (const auto &inputName : m_inputNames) {
            if (inputName.empty())
                throw std::runtime_error("Multiple inputs require naming!");


            unsigned index = connectionList.findOutput(inputName);
            if (index == ~0u)
                throw std::runtime_error(std::string("Can't find named output ") + inputName + " (yet)!");

            connectionList.outputs[index].consumer.push_back(this);
            m_inputs.push_back(index);
        }
    }

    for (const auto &outputName : m_outputNames) {
        ConvNet::Output output;
        output.name = outputName;
        output.producer = this;
        m_outputs.push_back(connectionList.outputs.size());
        connectionList.outputs.push_back(output);
    }
}




void Layer::loadInputOutputNames(tinyxml2::XMLElement *element)
{
    m_inputNames.clear();
    m_outputNames.clear();
    for (tinyxml2::XMLElement *elem = element->FirstChildElement("input"); elem != nullptr; elem = elem->NextSiblingElement("input")) {
        const char *name = elem->Attribute("name");
        if (name == nullptr)
            throw std::runtime_error("input tag needs 'name' attribute!");
        m_inputNames.push_back(name);
    }
    for (tinyxml2::XMLElement *elem = element->FirstChildElement("output"); elem != nullptr; elem = elem->NextSiblingElement("output")) {
        const char *name = elem->Attribute("name");
        if (name == nullptr)
            throw std::runtime_error("output tag needs 'name' attribute!");
        m_outputNames.push_back(name);
    }
}

void Layer::saveInputOutputNames(tinyxml2::XMLElement *element) const
{
    for (const auto &name : m_inputNames) {
        tinyxml2::XMLElement *elem  = element->GetDocument()->NewElement("input");
        elem->SetAttribute("name", name.c_str());
        element->LinkEndChild(elem);
    }
    for (const auto &name : m_outputNames) {
        tinyxml2::XMLElement *elem  = element->GetDocument()->NewElement("output");
        elem->SetAttribute("name", name.c_str());
        element->LinkEndChild(elem);
    }
}


void Layer::restoreSnapshot(ExecutionStream &/*stream*/, ConvNet::ConnectionList &connectionList, FileBlob &blob)
{
    std::uint32_t version;
    blob
        >> version
        >> m_name
        >> m_layerIndex
        >> m_parameterSetName
        >> m_inputs
        >> m_outputs
        >> m_inputNames
        >> m_outputNames;

    m_connectionList = &connectionList;
    for (unsigned i = 0; i < m_outputNames.size(); i++) {
        const auto &outputName = m_outputNames[i];

        ConvNet::Output output;
        output.name = outputName;
        output.producer = this;
        connectionList.outputs.push_back(output);
    }
}

void Layer::dumpSnapshot(FileBlob &blob) const
{
    std::uint32_t version = 1;
    blob
        << version
        << m_name
        << m_layerIndex
        << m_parameterSetName
        << m_inputs
        << m_outputs
        << m_inputNames
        << m_outputNames;
}

unsigned Layer::findInput(unsigned outputIndex) const
{
    for (unsigned i = 0; i < m_inputs.size(); i++)
        if (m_inputs[i] == outputIndex)
            return i;
    return ~0u;
}

unsigned Layer::findOutput(unsigned outputIndex) const
{
    for (unsigned i = 0; i < m_outputs.size(); i++)
        if (m_outputs[i] == outputIndex)
            return i;
    return ~0u;
}


}
