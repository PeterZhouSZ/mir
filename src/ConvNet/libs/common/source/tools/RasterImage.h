/*
 * Common Utilities - Distributed for "Mental Image Retrieval" implementation
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

#ifndef RASTERIMAGE_H
#define RASTERIMAGE_H

#include <vector>
#include <stdint.h>

#include <Eigen/Dense>

/** @addtogroup Codebase_Group
 *  @{
 */

class RasterImage
{
    public:
        RasterImage(unsigned w, unsigned h);
        RasterImage();
        ~RasterImage();

        void resize(unsigned w, unsigned h);

        void clear(uint32_t color);

        void writeToFile(const char *filename, unsigned quality = 100) const;
        void loadFromFile(const char *filename);
        static void getImageFileDimensions(const char *filename, unsigned &width, unsigned &height);

        void flipTopBottom();

        void drawLine(const Eigen::Vector2i &p1, const Eigen::Vector2i &p2, uint32_t color);
        void drawHLine(int x1, int x2, int y, uint32_t color);
        void drawVLine(int x, int y1, int y2, uint32_t color);
        void drawBox(const Eigen::Vector2i &p1, const Eigen::Vector2i &p2, uint32_t color, bool fill);
        void drawCircle(const Eigen::Vector2i &center, unsigned radius, uint32_t color);
        void drawFilledCircle(const Eigen::Vector2i &center, unsigned radius, uint32_t color);

        void mergeSideBySide(const RasterImage &left, const RasterImage &right, uint32_t clearColor);
        void mergeOnTopOfEachOther(const RasterImage &left, const RasterImage &right, uint32_t clearColor);

        uint32_t *getData() { return &m_pixelData[0]; }
        const uint32_t *getData() const { return &m_pixelData[0]; }
        
        void setPixel(unsigned i, unsigned r, unsigned g, unsigned b) { m_pixelData[i] = (r << 0) | (g << 8) | (b << 16) | 0xFF000000; }
        void setPixelXY(unsigned x, unsigned y, unsigned r, unsigned g, unsigned b) { m_pixelData[x+y*m_width] = (r << 0) | (g << 8) | (b << 16) | 0xFF000000; }
        unsigned &getPixelXY(unsigned x, unsigned y) { return m_pixelData[x+y*m_width]; }


        unsigned getWidth() const { return m_width; }
        unsigned getHeight() const { return m_height; }
    protected:
        std::vector<uint32_t> m_pixelData;
        unsigned m_width;
        unsigned m_height;

};

/**
 *  @}
 */


#endif // RASTERIMAGE_H
