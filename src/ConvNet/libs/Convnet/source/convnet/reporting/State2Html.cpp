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

#include "State2Html.h"

#include <boost/format.hpp>

namespace convnet {

void State2Html::addDefaultOutput(unsigned index, std::string label)
{
    if (index == ~0u) return;
    
    Output output;
    output.index = index;
    output.label = std::move(label);
    m_outputs.push_back(std::move(output));
}
    
    
void State2Html::produceTableHeader(std::ostream &stream)
{
    stream
        << "<table>" << std::endl
        << "  <tr>" << std::endl;
     
    for (const Output &output : m_outputs) {
        stream
            << "    <th>" << output.label << "</th>" << std::endl;
    }
        
    stream
        << "  </tr>" << std::endl;
        
    m_nextImageIndex = 0;
}

void State2Html::addTableRows(NetworkState &state, std::ostream &stream, HTTPReport &report, const std::vector<unsigned> &instances)
{
    for (const Output &output : m_outputs) {
        TensorData &tensorData = dynamic_cast<TensorData&>(*state.outputs[output.index]);
        std::vector<RasterImage> imgs = output.tensorRenderer.renderTensors(tensorData.getValues(), instances);

        for (unsigned j = 0; j < imgs.size(); j++)
            report.putImage(imgs[j], (boost::format("imgs/%s_output_%d_instance_%d.png") % m_imagePrefix % output.index % (m_nextImageIndex+j)).str());
    }

    for (unsigned j = 0; j < instances.size(); j++) {
        stream 
            << "  <tr>" << std::endl;
            
        for (const Output &output : m_outputs)
            stream << "    <td><img src=\"imgs/"<<m_imagePrefix<<"_output_"<<output.index<<"_instance_" << (m_nextImageIndex+j) << ".png\"/></td>" << std::endl;
        
        stream 
            << "  </tr>" << std::endl;
    }    

    m_nextImageIndex += instances.size();
}

void State2Html::produceTableFooter(std::ostream &stream)
{
    stream
         << "</table>" << std::endl;

}

}
