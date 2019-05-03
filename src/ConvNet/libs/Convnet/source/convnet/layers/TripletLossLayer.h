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

#ifndef TRIPLETLOSSLAYER_H
#define TRIPLETLOSSLAYER_H

#include "Layer.h"

namespace convnet {

class TripletLossLayer : public Layer
{
    public:
        static const char *TypeStr;
        
        TripletLossLayer(ConvNet &convnet, std::string name = std::string());

        virtual void save(tinyxml2::XMLElement *rootElement) const override;
        virtual void loadParameters(ConvNet &convnet, tinyxml2::XMLElement *element, Key<ConvNet>) override;
        
        virtual std::string toString() const override;

        virtual void resize(ExecutionStream &stream, Key<ConvNet>) override;
        
        inline TripletLossLayer *setWeight(float weight) { m_weight = weight; return this; }
        inline TripletLossLayer *setEvalOnly(bool evalOnly) { m_evalOnly = evalOnly; return this; }
    protected:
        float m_weight = 1.0f;
        float m_margin = 0.1f;
        bool m_evalOnly = false;
};

}

#endif // TRIPLETLOSSLAYER_H
