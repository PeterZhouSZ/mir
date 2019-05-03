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

#pragma once
#ifndef _TENSORRENDERER_H_
#define _TENSORRENDERER_H_

#include <cstdint>
#include <vector>

class RasterImage;

namespace convnet {

class Tensor;

class TensorRenderer {
    public:
        std::uint32_t backgroundColor = 0xFF000000;
        unsigned border = 4;
        unsigned separation = 2;
        
        unsigned upsample = 1;
        
        std::vector<unsigned> channelSelection;
        
        bool renderBinaryDifference = false;

        std::vector<RasterImage> renderTensors(Tensor &tensor, std::vector<unsigned> instances = std::vector<unsigned>({0})) const;
    protected:
};
    
}

#endif // _TENSORRENDERER_H_
