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

#include "Convnet.h"

#include "layers/Layer.h"
#include "FileBlob.h"

#include <tools/CPUStopWatch.h>

#include <boost/format.hpp>

#include <fstream>

#include <iostream>

namespace convnet {


LearnedParameters *ConvNet::ParameterSet::findByName(const std::string &name)
{
    for (auto &p : learnedParameters)
        if (p->getName() == name)
            return p.get();
    return nullptr;
}


void ConvNet::ConnectionList::addOutput(Output output)
{
    if (!output.name.empty()) {
        for (const auto &o : outputs)
            if (o.name == output.name)
                throw std::runtime_error("No two named outputs can have the same name!");
    }
    outputs.push_back(output);
}

unsigned ConvNet::ConnectionList::findOutput(const std::string &name) const
{
    for (unsigned i = 0; i < outputs.size(); i++)
        if (outputs[i].name == name)
            return i;
    return ~0u;
}

void ConvNet::clear()
{
    m_layers.clear();
    m_parameterSet.learnedParameters.clear();
    m_connectionList.outputs.clear();
}

void ConvNet::dropLastLayer()
{
    m_layers.pop_back();
}


void ConvNet::loadFromFile(const char *filename, bool append)
{
    if (!append)
        clear();
    
    tinyxml2::XMLDocument document;
    if (document.LoadFile(filename) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Could not load xml file!");

    tinyxml2::XMLElement *root = document.FirstChildElement("net");
    

    for (tinyxml2::XMLElement *layer = root->FirstChildElement("layer"); layer != nullptr; layer = layer->NextSiblingElement("layer"))
        addLayer(instantiateLayer(layer));
}

void ConvNet::saveToFile(const char *filename) const
{
    tinyxml2::XMLDocument document;

    tinyxml2::XMLElement *root = document.NewElement("net");
    document.LinkEndChild(root);

    for (const auto &layer : m_layers)
        layer->save(root);

    if (document.SaveFile(filename) != tinyxml2::XML_SUCCESS) throw std::runtime_error("Could not save xml file!");
}

void ConvNet::restoreSnapshot(ExecutionStream &stream, FileBlob &blob)
{
    clear();
    std::uint32_t version, numInstances, numParameters, numLayers;
    blob
        >> version
        >> numParameters
        >> numInstances
        >> numLayers;


    m_parameterSet.learnedParameters.reserve(numParameters);
    for (unsigned i = 0; i < numParameters; i++)
        restoreParameterSetFromBlob(blob);

    m_layers.reserve(numLayers);
    for (unsigned i = 0; i < numLayers; i++)
        restoreLayerFromBlob(blob);

    connectLayers();
    resizeLayers(stream, numInstances);
}


void ConvNet::connectLayers()
{
    m_connectionList.outputs.clear();
    for (unsigned i = 0; i < m_layers.size(); i++)
        m_layers[i]->connect(m_connectionList, {});
}

void ConvNet::resizeLayers(ExecutionStream &stream, unsigned numInstances)
{
    m_numInstances = numInstances;
    for (unsigned i = 0; i < m_layers.size(); i++) {
		try {
	        m_layers[i]->resize(stream, {});
		} catch (const std::exception &e) {
			std::cout << "Error occured resizing layer " << m_layers[i]->getName() << "  " << m_layers[i]->toString() << std::endl;
			throw;
		}
	}
}


NetworkState ConvNet::allocateState(ExecutionStream &stream, bool forward, bool backward, bool parameterUpdate) const
{
    NetworkState state;
    state.outputs.resize(m_connectionList.outputs.size());
    state.auxLayerData.resize(m_layers.size());
    for (unsigned i = 0; i < m_layers.size(); i++)
        m_layers[i]->allocateState(state, stream, forward, backward, parameterUpdate, {});
    return state;
}


void ConvNet::feedForward(NetworkState &state, ExecutionWorkspace &workspace, ExecutionStream &stream, bool trainingMode) const
{
    for (unsigned i = 0; i < m_layers.size(); i++)
        m_layers[i]->forward(state, workspace, stream, trainingMode);
}


void ConvNet::feedBackward(NetworkState &state, ExecutionWorkspace &workspace, ExecutionStream &stream, bool accumulateParameterGradients) const
{
    for (unsigned i = 0; i < m_layers.size(); i++) {
        unsigned revIdx = m_layers.size()-1-i;

        m_layers[revIdx]->backward(state, workspace, stream, accumulateParameterGradients);
    }
}

void ConvNet::feedForwardEx(NetworkState &state, ExecutionWorkspace &workspace, ExecutionStream &stream,
                    bool trainingMode, const std::function<bool(Layer&)> &enabled) const
{
    for (unsigned i = 0; i < m_layers.size(); i++)
        if (enabled(*m_layers[i]))
            m_layers[i]->forward(state, workspace, stream, trainingMode);
}

void ConvNet::feedBackwardEx(NetworkState &state, ExecutionWorkspace &workspace, ExecutionStream &stream,
                    const std::function<bool(Layer&)> &enabled, const std::function<bool(Layer&)> &accumulateParameterGradients) const
{
    for (unsigned i = 0; i < m_layers.size(); i++) {
        unsigned revIdx = m_layers.size()-1-i;

        if (enabled(*m_layers[revIdx]))
            m_layers[revIdx]->backward(state, workspace, stream, accumulateParameterGradients(*m_layers[revIdx]));
    }
}


void ConvNet::benchmarkForward(NetworkState &state, ExecutionWorkspace &workspace, ExecutionStream &stream)
{
    std::vector<std::pair<std::string, float>> times;
    float totalTime = 0.0f;

    stream.flush();
    for (unsigned i = 0; i < m_layers.size(); i++) {
        Engine::CPUStopWatch timer;
        for (unsigned j = 0; j < 10; j++) {
            m_layers[i]->forward(state, workspace, stream, false);
        }
        stream.flush();
        float time = timer.getNanoseconds() / 10.0f * 1e-9f;
        times.push_back({
            (boost::format("layer %i %s %s") % i % m_layers[i]->getName() % m_layers[i]->toString()).str(),
            time
        });
        totalTime += time;
    }
    boost::io::ios_flags_saver ifs(std::cout);
    for (const auto &p : times) {
        std::cout << std::setw(5) << std::fixed << std::setprecision(4) << (p.second / totalTime * 100) << "%   " << std::setw(10) << p.second << " [s] " << p.first << std::endl;
    }
}

void ConvNet::addLayer(Layer *layer)
{
    layer->setLayerIndex(m_layers.size(), {});
    m_layers.push_back(std::unique_ptr<Layer>(layer));
}

void ConvNet::insertLayer(unsigned insertBefore, Layer *layer)
{
    m_layers.insert(m_layers.begin()+insertBefore, std::unique_ptr<Layer>(layer));
    for (unsigned i = insertBefore; i < m_layers.size(); i++)
        m_layers[i]->setLayerIndex(i, {});
}

void ConvNet::dropLayer(unsigned index)
{
    m_layers.erase(m_layers.begin()+index);
    for (unsigned i = index; i < m_layers.size(); i++)
        m_layers[i]->setLayerIndex(i, {});
}


Layer *ConvNet::getLayer(const std::string &name)
{
    for (auto &l : m_layers)
        if (l->getName() == name)
            return l.get();

    return nullptr;
}

void ConvNet::pushParametersToBackend(ExecutionWorkspace &workspace, ExecutionStream &stream)
{
    for (auto &p : m_parameterSet.learnedParameters)
        p->pushToBackend(workspace, stream);
}

void ConvNet::pullParametersFromBackend(ExecutionWorkspace &workspace, ExecutionStream &stream)
{
    for (auto &p : m_parameterSet.learnedParameters)
        p->pullFromBackend(workspace, stream);
}

void ConvNet::performParameterStep(float stepsize, ExecutionWorkspace &workspace, ExecutionStream &stream)
{
    for (auto &p : m_parameterSet.learnedParameters)
        p->performParameterStep(stepsize, workspace, stream);
}

void ConvNet::performParameterStepEx(float stepsize, ExecutionWorkspace &workspace, ExecutionStream &stream,
                                    const std::function<bool(LearnedParameters&)> &enabled)
{
    for (auto &p : m_parameterSet.learnedParameters)
        if (enabled(*p))
            p->performParameterStep(stepsize, workspace, stream);
}

Layer *ConvNet::instantiateLayer(tinyxml2::XMLElement *element)
{
    const char *layerType = element->Attribute("type");
    if (layerType == nullptr) throw std::runtime_error("Missing type attribute!");

    const char *layerName = element->Attribute("name");
    std::string layerNameStr;
    if (layerName != nullptr)
        layerNameStr = layerName;

    std::string layerTypeStr(layerType);

    Layer *layer = instantiateLayer(std::move(layerNameStr), layerTypeStr);
    try {
        layer->loadParameters(*this, element, {});
    } catch (...) {
        delete layer;
        throw;
    }
    return layer;
}

Layer *ConvNet::addLayer(std::string name, const std::string &typeStr, std::string parameters) { 
    Layer *layer = instantiateLayer(std::move(name), typeStr); 
    addLayer(layer); 
    layer->setupParameters(*this, std::move(parameters), {});
    return layer;
}

Layer *ConvNet::insertLayer(unsigned insertBefore, std::string name, const std::string &typeStr, std::string parameters) { 
    Layer *layer = instantiateLayer(std::move(name), typeStr); 
    insertLayer(insertBefore, layer); 
    layer->setupParameters(*this, std::move(parameters), {});
    return layer;
}

}
