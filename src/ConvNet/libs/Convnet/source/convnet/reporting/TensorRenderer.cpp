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

#include "TensorRenderer.h"

#include "../Tensor.h"

#include <tools/RasterImage.h>

namespace convnet {


std::vector<RasterImage> TensorRenderer::renderTensors(Tensor &tensor, std::vector<unsigned> instances) const
{
    std::vector<unsigned> channels = channelSelection;
    if (channels.empty())
        for (unsigned i = 0; i < tensor.getSize().channels; i++)
            channels.push_back(i);
    
    while (channels.size() % 3 != 0)
        channels.push_back(channels.back());
        
    const unsigned numCols = (channels.size()+2)/3;
    const unsigned numRows = tensor.getSize().depth;
    
    std::vector<RasterImage> imgs;
    imgs.resize(instances.size());
    
    for (auto &img : imgs) {
        img.resize(
            2*border + numCols * tensor.getSize().width * upsample + (numCols-1) * separation,
            2*border + numRows * tensor.getSize().height * upsample + (numRows-1) * separation
        );
        img.clear(backgroundColor);
    }
    
    MappedTensor mappedTensor = tensor.lock();
    
    for (unsigned i = 0; i < instances.size(); i++) {
        const unsigned instance = instances[i];
        
        for (unsigned row = 0; row < numRows; row++) {
            for (unsigned col = 0; col < numCols; col++) {
                for (unsigned y = 0; y < tensor.getSize().height; y++)
                    for (unsigned x = 0; x < tensor.getSize().width; x++) {

                        if (!renderBinaryDifference) {
                            int r = std::min<int>(std::max<int>(mappedTensor.get<float>(x, y, row, channels[col*3+0], instance) * 127.0f/3.0f + 127.0f, 0), 255);
                            int g = std::min<int>(std::max<int>(mappedTensor.get<float>(x, y, row, channels[col*3+1], instance) * 127.0f/3.0f + 127.0f, 0), 255);
                            int b = std::min<int>(std::max<int>(mappedTensor.get<float>(x, y, row, channels[col*3+2], instance) * 127.0f/3.0f + 127.0f, 0), 255);
                            
                            unsigned color = 
                                        (r << 0) |
                                        (g << 8) |
                                        (b << 16) |
                                        (0xFF << 24);
                            
                            for (unsigned k = 0; k < upsample; k++)
                                for (unsigned l = 0; l < upsample; l++)
                                    imgs[i].getData()[
                                            (border + col * (separation + tensor.getSize().width*upsample) + x*upsample+l) +
                                            (border + row * (separation + tensor.getSize().height*upsample) + y*upsample+k) * imgs[i].getWidth()] = color;
                        } else {
							float diff;
							if (tensor.getSize().channels == 1)
								diff = mappedTensor.get<float>(x, y, row, channels[col*3+0], instance);
							else
                            	diff = mappedTensor.get<float>(x, y, row, channels[col*3+0], instance) - mappedTensor.get<float>(x, y, row, channels[col*3+1], instance);
                            int r = std::min<int>(std::max<int>(diff * 255.0f/3.0f, 0), 255);
                            int g = std::min<int>(std::max<int>(-diff * 255.0f/3.0f, 0), 255);
                            
                            unsigned color = 
                                        (r << 0) |
                                        (g << 8) |
                                        (0xFF << 24);

                            for (unsigned k = 0; k < upsample; k++)
                                for (unsigned l = 0; l < upsample; l++)
                                    imgs[i].getData()[
                                            (border + col * (separation + tensor.getSize().width*upsample) + x*upsample+l) +
                                            (border + row * (separation + tensor.getSize().height*upsample) + y*upsample+k) * imgs[i].getWidth()] = color;
                        }
                    }
            }    
        }    
    }
    
    tensor.unlock(mappedTensor, false);
    
    return imgs;
}

}

