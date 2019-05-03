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

#ifndef STATE2HTML_H
#define STATE2HTML_H

#include "HTTPReport.h"
#include "TensorRenderer.h"

#include "../NetworkState.h"

namespace convnet {

class State2Html
{
    public:
        struct Output {
            unsigned index;
            std::string label;
            TensorRenderer tensorRenderer;
        };
        
        State2Html(std::string prefix) : m_imagePrefix(std::move(prefix)) { }
        
        void addDefaultOutput(unsigned index, std::string label);
        
        void produceTableHeader(std::ostream &stream);
        void addTableRows(NetworkState &state, std::ostream &stream, HTTPReport &report, const std::vector<unsigned> &instances);
        void produceTableFooter(std::ostream &stream);
        
        inline std::vector<Output> &getOutputs() { return m_outputs; }
    protected:
        std::string m_imagePrefix;
        std::vector<Output> m_outputs;
        unsigned m_nextImageIndex;
        
};

}

#endif // STATE2HTML_H
