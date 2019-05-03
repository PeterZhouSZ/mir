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

#ifndef INPUTLAYER_H
#define INPUTLAYER_H

#include "Layer.h"

namespace convnet {

class InputLayer : public Layer
{
    public:
        static const char *TypeStr;

        InputLayer(ConvNet &convnet, std::string name = std::string());

        virtual InputStagingBuffer allocateStagingBuffer(ExecutionStream &stream) const = 0;
        void swapInStagingBuffer(NetworkState &state, InputStagingBuffer &stagingBuffer) const;

        virtual bool consumesOutputGradients(unsigned /*output*/) const override { return false; }
        
        virtual InputLayer* setInputSizes(std::vector<TensorSize> inputSizes);

        virtual void save(tinyxml2::XMLElement *rootElement) const override;

        virtual std::string toString() const override;

        virtual void resize(ExecutionStream &stream, Key<ConvNet>) override;
    protected:
        std::vector<TensorSize> m_inputSizes;
};

}

#endif // INPUTLAYER_H
