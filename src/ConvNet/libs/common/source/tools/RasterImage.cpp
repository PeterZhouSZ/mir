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

#include "RasterImage.h"

#include <Magick++.h>
#include <stdexcept>
#include <string.h>

#include <stdlib.h>

RasterImage::RasterImage(unsigned w, unsigned h)
{
    resize(w, h);
}

RasterImage::RasterImage()
{
    resize(0, 0);
}


RasterImage::~RasterImage()
{
    //dtor
}

void RasterImage::resize(unsigned w, unsigned h)
{
    m_pixelData.resize(w*h);
    m_width = w;
    m_height = h;
}


void RasterImage::clear(uint32_t color)
{
    for (unsigned i = 0; i < m_pixelData.size(); i++)
        m_pixelData[i] = color;
}


void RasterImage::writeToFile(const char *filename, unsigned quality) const
{
    Magick::Image image(m_width, m_height, "RGBA", Magick::CharPixel, &m_pixelData[0]);
    image.quality(quality);
    image.write(filename);
}

void RasterImage::loadFromFile(const char *filename)
{
    Magick::Image image(filename);


    if (image.depth() != 8)
        throw std::runtime_error("Invalid bit depth.");

    unsigned width = image.columns();
    unsigned height = image.rows();
/*
    switch (image.type()) {
        case Magick::TrueColorType:
        break;
        case Magick::TrueColorMatteType:
        break;
        default:
            throw std::runtime_error("only true color images are supported.");
    }
*/
    resize(width, height);
    image.write(0, 0, image.columns(), image.rows(), "RGBA", Magick::CharPixel, &m_pixelData[0]);
}

void RasterImage::getImageFileDimensions(const char *filename, unsigned &width, unsigned &height)
{
    Magick::Image image(filename);

    width = image.columns();
    height = image.rows();
}

void RasterImage::flipTopBottom()
{
    std::vector<uint32_t> tmp;
    tmp.resize(m_width);
    for (unsigned i = 0; i < m_height/2; i++) {
        memcpy(tmp.data(), &m_pixelData[i*m_width], m_width*4);
        memcpy(&m_pixelData[i*m_width], &m_pixelData[(m_height-1-i)*m_width], m_width*4);
        memcpy(&m_pixelData[(m_height-1-i)*m_width], tmp.data(), m_width*4);
    }
}



void RasterImage::drawLine(const Eigen::Vector2i &p1, const Eigen::Vector2i &p2, uint32_t color)
{

    Eigen::Vector2i D = p2 - p1;
    Eigen::Vector2i startPoint;
    Eigen::Vector2i endPoint;

    if (abs(D[0]) > abs(D[1])) {

        if (p1[0] > p2[0]) {
            startPoint = p2;
            endPoint = p1;
            D[0] = -D[0];
            D[1] = -D[1];
        } else {
            startPoint = p1;
            endPoint = p2;
        }

        if (D[1] > 0) {
            int error = 0;
            int y = startPoint[1];
            for (int i = startPoint[0]; i <= endPoint[0]; i++) {
                if ((i >= 0) && (i < (int)m_width) &&
                    (y >= 0) && (y < (int)m_height))
                    m_pixelData[i + y*m_width] = color;

                error += D[1];
                if (error >= D[0]) {
                    y++;
                    error -= D[0];
                }
            }
        } else {
            int error = 0;
            int y = startPoint[1];
            for (int i = startPoint[0]; i <= endPoint[0]; i++) {
                if ((i >= 0) && (i < (int)m_width) &&
                    (y >= 0) && (y < (int)m_height))
                    m_pixelData[i + y*m_width] = color;

                error -= D[1];
                if (error >= D[0]) {
                    y--;
                    error -= D[0];
                }
            }
        }
    } else {
        if (p1[1] > p2[1]) {
            startPoint = p2;
            endPoint = p1;
            D[0] = -D[0];
            D[1] = -D[1];
        } else {
            startPoint = p1;
            endPoint = p2;
        }
        if (D[0] > 0) {
            int error = 0;
            int x = startPoint[0];
            for (int i = startPoint[1]; i <= endPoint[1]; i++) {
                if ((i >= 0) && (i < (int)m_height) &&
                    (x >= 0) && (x < (int)m_width))
                    m_pixelData[x + i*m_width] = color;

                error += D[0];
                if (error >= D[1]) {
                    x++;
                    error -= D[1];
                }
            }
        } else {
            int error = 0;
            int x = startPoint[0];
            for (int i = startPoint[1]; i <= endPoint[1]; i++) {
                if ((i >= 0) && (i < (int)m_height) &&
                    (x >= 0) && (x < (int)m_width))
                    m_pixelData[x + i*m_width] = color;

                error -= D[0];
                if (error >= D[1]) {
                    x--;
                    error -= D[1];
                }
            }
        }
    }

}

void RasterImage::drawHLine(int x1, int x2, int y, uint32_t color)
{
    if (x1 >= (int)m_width)
        return;
    if (x2 < 0)
        return;

    if (y < 0)
        return;
    if (y >= (int)m_height)
        return;

    x1 = std::max(0, x1);
    x2 = std::min<int>(m_width-1, x2);
    for (int x = x1; x <= x2; x++)
        m_pixelData[x+y*m_width] = color;
}

void RasterImage::drawVLine(int x, int y1, int y2, uint32_t color)
{
    if (x >= (int)m_width)
        return;
    if (x < 0)
        return;

    if (y1 >= (int)m_height)
        return;
    if (y2 < 0)
        return;

    y1 = std::max(0, y1);
    y2 = std::min<int>(m_height-1, y2);

    for (int y = y1; y <= y2; y++)
        m_pixelData[x+y*m_width] = color;
}


void RasterImage::drawBox(const Eigen::Vector2i &p1, const Eigen::Vector2i &p2, uint32_t color, bool fill)
{
    if (p2[0] < 0)
        return;
    if (p2[1] < 0)
        return;

    if (p1[0] >= (int)m_width)
        return;
    if (p1[1] >= (int)m_height)
        return;


    if (fill) {
        int miX = std::max(0, p1[0]);
        int maX = std::min<int>(m_width-1, p2[0]);

        int miY = std::max(0, p1[1]);
        int maY = std::min<int>(m_height-1, p2[1]);

        for (int y = miY; y <= maY; y++)
            for (int x = miX; x <= maX; x++)
                m_pixelData[x+y*m_width] = color;
    } else {
        drawHLine(p1[0], p2[0], p1[1], color);
        drawHLine(p1[0], p2[0], p2[1], color);

        drawVLine(p1[0], p1[1], p2[1], color);
        drawVLine(p2[0], p1[1], p2[1], color);
    }
}


void RasterImage::drawCircle(const Eigen::Vector2i &center, unsigned radius, uint32_t color)
{
    int sqrR = (radius+1)*(radius+1);
    int x = radius;
    int y = 0;
    int sqrX = x * x;
    while (x >= y) {

#define PutPixel(px_, py_)                                        \
        {                                                         \
            int px = px_;                                         \
            int py = py_;                                         \
            if ((px >= 0) && (px < (int)m_width) &&                    \
                (py >= 0) && (py < (int)m_height))                     \
                m_pixelData[px + py*m_width] = color;       \
        }

        PutPixel(center[0] + x, center[1] + y);
        PutPixel(center[0] + y, center[1] + x);

        PutPixel(center[0] - x, center[1] + y);
        PutPixel(center[0] - y, center[1] + x);

        PutPixel(center[0] - x, center[1] - y);
        PutPixel(center[0] - y, center[1] - x);

        PutPixel(center[0] + x, center[1] - y);
        PutPixel(center[0] + y, center[1] - x);


        y++;
        if (sqrX + y*y > sqrR) {
            x--;
            sqrX = x*x;
        }
    }
}

void RasterImage::drawFilledCircle(const Eigen::Vector2i &center, unsigned radius, uint32_t color)
{
    int y0 = center[1] - (int)radius;
    int y1 = center[1] + (int)radius;

    if (y1 < 0)
        return;
    if (y0 >= (int) m_height)
        return;

    y0 = std::max(y0, 0);
    y1 = std::min(y1, (int) m_height - 1);

    int sqrR = radius*radius;

    for (int y = y0; y <= y1; y++) {
        int dy = y-center[1];
        int dx = std::sqrt(std::max(0, sqrR - dy*dy));
        int x0 = center[0] - dx;
        int x1 = center[0] + dx;
        if (x0 >= (int) m_width)
            continue;
        if (x1 < 0)
            continue;

        x0 = std::max(x0, 0);
        x1 = std::min(x1, (int) m_width - 1);

        for (int x = x0; x <= x1; x++) {
            m_pixelData[y*m_width+x] = color;
        }
    }
}


void RasterImage::mergeSideBySide(const RasterImage &left, const RasterImage &right, uint32_t clearColor)
{
    resize(left.getWidth()+right.getWidth(), std::max(left.getHeight(), right.getHeight()));

    for (unsigned i = 0; i < m_width*m_height; i++)
        m_pixelData[i] = clearColor;

    for (unsigned i = 0; i < left.getHeight(); i++)
        memcpy(&m_pixelData[i*m_width], &left.getData()[i*left.getWidth()], left.getWidth()*4);

    for (unsigned i = 0; i < right.getHeight(); i++)
        memcpy(&m_pixelData[i*m_width + left.getWidth()], &right.getData()[i*right.getWidth()], right.getWidth()*4);

}

void RasterImage::mergeOnTopOfEachOther(const RasterImage &top, const RasterImage &bottom, uint32_t clearColor)
{
    resize(std::max(top.getWidth(), bottom.getWidth()), top.getHeight() + bottom.getHeight());

    for (unsigned i = 0; i < m_width*m_height; i++)
        m_pixelData[i] = clearColor;

    for (unsigned i = 0; i < top.getHeight(); i++)
        memcpy(&m_pixelData[i*m_width], &top.getData()[i*top.getWidth()], top.getWidth()*4);

    for (unsigned i = 0; i < bottom.getHeight(); i++)
        memcpy(&m_pixelData[(i + top.getHeight())*m_width], &bottom.getData()[i*bottom.getWidth()], bottom.getWidth()*4);

}
