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

#ifndef LAYER_H
#define LAYER_H

#include <tinyxml2.h>

#include <string>
#include <vector>

#include "../NetworkState.h"
#include "../Convnet.h"
#include "../ExecutionWorkspace.h"
#include "../ExecutionStream.h"

namespace convnet {

template <typename T>
class Key { friend T; Key() {} };

class ConvNet;
class Layer;


class LearnedParameters {
    public:
        LearnedParameters(std::string name);
        virtual ~LearnedParameters() = default;

        inline const std::string &getName() const { return m_name; }

        virtual void performParameterStep(float stepsize, ExecutionWorkspace &workspace, ExecutionStream &stream) = 0;
        virtual void pushToBackend(ExecutionWorkspace &workspace, ExecutionStream &stream) = 0;
        virtual void pullFromBackend(ExecutionWorkspace &workspace, ExecutionStream &stream) = 0;

        virtual float computeRegularizationLoss(ExecutionWorkspace &workspace, ExecutionStream &stream) = 0;
        
        virtual void restoreSnapshot(ExecutionStream &stream, FileBlob &blob) = 0;
        virtual void dumpSnapshot(ExecutionWorkspace &workspace, ExecutionStream &stream, FileBlob &blob) = 0;
    protected:
        std::string m_name;
};

class Layer
{
    public:
        Layer(ConvNet &convnet, std::string name = std::string());
        virtual ~Layer() = default;
        
        virtual void setupParameters(ConvNet &/*convnet*/, std::string /*parameterName*/, Key<ConvNet>) { }
        virtual void loadParameters(ConvNet &/*convnet*/, tinyxml2::XMLElement *element, Key<ConvNet>) { loadInputOutputNames(element); }

        virtual void restoreSnapshot(ExecutionStream &stream, ConvNet::ConnectionList &connectionList, FileBlob &blob);
        virtual void dumpSnapshot(FileBlob &blob) const;
        
        virtual bool consumesOutputGradients(unsigned /*output*/) const { return true; }
        virtual bool consumesInputValues(unsigned /*input*/) const { return true; }

        inline const std::string &getName() const { return m_name; }

        inline void setLayerIndex(unsigned layerIndex, Key<ConvNet>) { m_layerIndex = layerIndex; }
        inline unsigned getLayerIndex() const { return m_layerIndex; }

        inline Layer* setInputConnectionNames(std::vector<std::string> inputNames) { m_inputNames = std::move(inputNames); return this; }
        inline Layer* setOutputConnectionNames(std::vector<std::string> outputNames) { m_outputNames = std::move(outputNames); return this; }

        void connect(ConvNet::ConnectionList &connectionList, Key<ConvNet>);

        virtual void resize(ExecutionStream &stream, Key<ConvNet>) = 0;
        virtual void allocateState(NetworkState &networkState, ExecutionStream &stream, bool forward, bool backward, bool parameterUpdate, Key<ConvNet>) const = 0;

        virtual void forward(NetworkState &networkState, ExecutionWorkspace &workspace, ExecutionStream &stream, bool trainingMode) const = 0;
        virtual void backward(NetworkState &networkState, ExecutionWorkspace &workspace, ExecutionStream &stream, bool accumulateParameterGradients) const = 0;

        virtual void save(tinyxml2::XMLElement *rootElement) const = 0;
        
        virtual std::size_t getWorkspaceSize() { return 0; }

        virtual std::string toString() const = 0;

        inline NetworkData &getInput(NetworkState &networkState, unsigned inputIdx = 0) { return *networkState.outputs[m_inputs[inputIdx]]; }
        inline const NetworkData &getInput(const NetworkState &networkState, unsigned inputIdx = 0) const { return *networkState.outputs[m_inputs[inputIdx]]; }
        inline NetworkData &getOutput(NetworkState &networkState, unsigned outputIdx = 0) { return *networkState.outputs[m_outputs[outputIdx]]; }
        inline const NetworkData &getOutput(const NetworkState &networkState, unsigned outputIdx = 0) const { return *networkState.outputs[m_outputs[outputIdx]]; }

        inline unsigned getInputIdx(unsigned input = 0) const { return m_inputs[input]; }
        inline unsigned getOutputIdx(unsigned output = 0) const { return m_outputs[output]; }

        inline unsigned getNumInputs() const { return m_inputNames.size(); }
        inline unsigned getNumOutputs() const { return m_outputNames.size(); }
        
        unsigned findInput(unsigned outputIndex) const;
        unsigned findOutput(unsigned outputIndex) const;

    protected:
        ConvNet &m_convnet;
        std::string m_name;
        unsigned m_layerIndex;

        std::string m_parameterSetName;
        ConvNet::ConnectionList *m_connectionList = nullptr;
        std::vector<unsigned> m_inputs;
        std::vector<unsigned> m_outputs;

        std::vector<std::string> m_inputNames;
        std::vector<std::string> m_outputNames;

        void loadInputOutputNames(tinyxml2::XMLElement *element);
        void saveInputOutputNames(tinyxml2::XMLElement *element) const;
};

}

#endif // LAYER_H
