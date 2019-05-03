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

#include "TripletLossLayer.h"


namespace convnet {

const char *TripletLossLayer::TypeStr = "triplet_loss";

TripletLossLayer::TripletLossLayer(ConvNet &convnet, std::string name) : Layer(convnet, std::move(name))
{
    m_outputNames.push_back("");
}

void TripletLossLayer::save(tinyxml2::XMLElement *rootElement) const
{
    tinyxml2::XMLElement *element = rootElement->GetDocument()->NewElement("layer");
    rootElement->LinkEndChild(element);

    element->SetAttribute("type", TypeStr);
    element->SetAttribute("weight", m_weight);
    element->SetAttribute("name", m_name.c_str());
    if (m_evalOnly)
        element->SetAttribute("eval_only", "true");

    saveInputOutputNames(element);
}

void TripletLossLayer::loadParameters(ConvNet &/*convnet*/, tinyxml2::XMLElement *element, Key<ConvNet>)
{
    loadInputOutputNames(element);

    if (element->QueryAttribute("weight", &m_weight) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Error reading TripletLossLayer weight!");


    const char *lossType = element->Attribute("loss_type");
    if (lossType == nullptr)
        throw std::runtime_error("No mode attribute found!");

    const char *evalOnly = element->Attribute("eval_only");
    if (lossType == nullptr)
        m_evalOnly = false;
    else 
        m_evalOnly = std::string(evalOnly) == "true";
}

std::string TripletLossLayer::toString() const
{
    std::stringstream str;
    str << TypeStr << " on " << m_connectionList->outputs[m_inputs[0]].size;
    return str.str();
}

void TripletLossLayer::resize(ExecutionStream &/*stream*/, Key<ConvNet>)
{
    if (m_inputs.size() != 3)
        throw std::runtime_error("TripletLossLayer inputs must be 3");

    if (m_outputs.size() != 1)
        throw std::runtime_error("TripletLossLayer outputs must be 1");


    TensorSize &outputSize = m_connectionList->outputs[m_outputs[0]].size;
    
    outputSize = m_connectionList->outputs[m_inputs[0]].size;
    outputSize.channels = 1;
    
    for (unsigned i : m_inputs) {
        const TensorSize &inputSize0 = m_connectionList->outputs[m_inputs[0]].size;
        const TensorSize &inputSize = m_connectionList->outputs[i].size;
        
        if (inputSize != inputSize0) throw std::runtime_error("Input sizes don't match!");
    }

}

}
