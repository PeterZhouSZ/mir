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

#ifndef POOLINGLAYER_H
#define POOLINGLAYER_H

#include "Layer.h"

namespace convnet {

class PoolingLayer : public Layer
{
    public:
        static const char *TypeStr;

        enum Mode {
            MODE_AVG,
            MODE_MAX
        };

        PoolingLayer(ConvNet &convnet, std::string name = std::string());

        virtual void save(tinyxml2::XMLElement *rootElement) const override;

        virtual std::string toString() const override;

        PoolingLayer *setWindowSize(unsigned w, unsigned h, unsigned d);
        PoolingLayer *setMode(Mode mode) { m_mode = mode; structureChanged(); return this; }

        virtual void loadParameters(ConvNet &convnet, tinyxml2::XMLElement *element, Key<ConvNet>) override;

        virtual void resize(ExecutionStream &stream, Key<ConvNet>) override;

        virtual void structureChanged() { }
    protected:
        unsigned m_windowSize[3];
        Mode m_mode = MODE_AVG;
};

}

#endif // POOLINGLAYER_H
