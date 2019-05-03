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

#ifndef CONVNET_H
#define CONVNET_H

#include "Tensor.h"
#include "NetworkState.h"
#include "ExecutionWorkspace.h"
#include "ExecutionStream.h"


#include <tools/TaskScheduler.h>

#include <mutex>
#include <thread>
#include <string>
#include <map>

#include <tinyxml2.h>


namespace convnet {

class LearnedParameters;
class Layer;
class ConvolutionLayer;
class FileBlob;

class ConvNet
{
    public:
        struct Output {
            std::string name;
            TensorSize size;
            Layer *producer = nullptr;
            std::vector<Layer*> consumer;
        };

        struct ConnectionList {
            std::vector<Output> outputs;
            void addOutput(Output output);
            unsigned findOutput(const std::string &name) const;
        };

        struct ParameterSet {
            std::vector<std::unique_ptr<LearnedParameters>> learnedParameters;

            LearnedParameters *findByName(const std::string &name);
        };

        inline const Layer *getLayer(unsigned idx) const { return m_layers[idx].get(); }
        inline Layer *getLayer(unsigned idx) { return m_layers[idx].get(); }
        inline unsigned getNumLayer() const { return m_layers.size(); }

        void clear();

        void dropLastLayer();

        void connectLayers();
        void resizeLayers(ExecutionStream &stream, unsigned numInstances);

        void loadFromFile(const char *filename, bool append = false);
        void saveToFile(const char *filename) const;

        void restoreSnapshot(ExecutionStream &stream, FileBlob &blob);
        void dumpSnapshot(ExecutionWorkspace &workspace, ExecutionStream &stream, FileBlob &blob) const;

        virtual std::unique_ptr<ExecutionWorkspace> allocateExecutionWorkspace(bool forward = true, bool backward = true, bool parameterUpdate = true) = 0;
        virtual std::unique_ptr<ExecutionStream> allocateExecutionStream() = 0;
        virtual std::unique_ptr<ExecutionStreamWaitingFence> allocateWaitFence() = 0;
        virtual std::unique_ptr<ExecutionStreamSyncFence> allocateSyncFence() = 0;


        NetworkState allocateState(ExecutionStream &stream, bool forward = true, bool backward = true, bool parameterUpdate = true) const;

        void feedForward(NetworkState &state, ExecutionWorkspace &workspace, ExecutionStream &stream, bool trainingMode) const;
        void feedBackward(NetworkState &state, ExecutionWorkspace &workspace, ExecutionStream &stream, bool accumulateParameterGradients) const;

        void feedForwardEx(NetworkState &state, ExecutionWorkspace &workspace, ExecutionStream &stream,
                           bool trainingMode, const std::function<bool(Layer&)> &enabled) const;
        void feedBackwardEx(NetworkState &state, ExecutionWorkspace &workspace, ExecutionStream &stream,
                            const std::function<bool(Layer&)> &enabled, const std::function<bool(Layer&)> &accumulateParameterGradients) const;

        void pushParametersToBackend(ExecutionWorkspace &workspace, ExecutionStream &stream);
        void pullParametersFromBackend(ExecutionWorkspace &workspace, ExecutionStream &stream);
        void performParameterStep(float stepsize, ExecutionWorkspace &workspace, ExecutionStream &stream);
        void performParameterStepEx(float stepsize, ExecutionWorkspace &workspace, ExecutionStream &stream,
                                    const std::function<bool(LearnedParameters&)> &enabled);

        void benchmarkForward(NetworkState &state, ExecutionWorkspace &workspace, ExecutionStream &stream);

        inline ParameterSet &getParameterSet() { return m_parameterSet; }

        Layer *getLayer(const std::string &name);

        template<class LayerType>
        LayerType *getLayer(const std::string &name) {
            return dynamic_cast<LayerType*>(getLayer(name));
        }

        inline const ConnectionList &getConnectionList() const { return m_connectionList; }

        Layer *addLayer(std::string name, const std::string &typeStr, std::string parameters);
        
        template<class LayerType>
        LayerType *addLayer(std::string name = std::string(), std::string parameters = std::string()) {
            return dynamic_cast<LayerType*>(addLayer(std::move(name), LayerType::TypeStr, std::move(parameters)));
        }

        Layer *insertLayer(unsigned insertBefore, std::string name, const std::string &typeStr, std::string parameters);        
        template<class LayerType>
        LayerType *insertLayer(unsigned insertBefore, std::string name = std::string(), std::string parameters = std::string()) {
            return dynamic_cast<LayerType*>(insertLayer(insertBefore, std::move(name), LayerType::TypeStr, std::move(parameters)));
        }
        
        void dropLayer(unsigned index);
        
        
        inline unsigned getNumInstances() const { return m_numInstances; }
    protected:
        void addLayer(Layer *layer);
        void insertLayer(unsigned insertBefore, Layer *layer);

        virtual void restoreLayerFromBlob(FileBlob &blob) = 0;
        virtual void restoreParameterSetFromBlob(FileBlob &blob) = 0;

        virtual Layer *instantiateLayer(tinyxml2::XMLElement *element);
        virtual Layer *instantiateLayer(std::string name, const std::string &type) = 0;

        std::vector<std::unique_ptr<Layer>> m_layers;
        ConnectionList m_connectionList;
        ParameterSet m_parameterSet;

        unsigned m_numInstances;
};

}

#endif // GENERATIVECONVNET_H
