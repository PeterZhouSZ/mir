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


#ifndef SVGCOMPOSER_H
#define SVGCOMPOSER_H

#include <ostream>

#include "RasterImage.h"

/** @addtogroup Codebase_Group
 *  @{
 */

class SvgComposer
{
    public:
        SvgComposer(std::ostream &stream, unsigned width, unsigned height);
        ~SvgComposer();

        class Image {
            public:
                enum Format {
                    FORMAT_JPG,
                    FORMAT_PNG
                };

                Image(const RasterImage &image, float x = 0.0f, float y = 0.0f);

                Image &setFormat(Format format) { m_format = format; return *this; }
                Image &setJpgQuality(unsigned quality) { m_jpgQuality = quality; return *this; }
                Image &rescale(unsigned w, unsigned h) { m_rescaleWidth = w; m_rescaleHeight = h; return *this; }
                Image &setWidthHeight(unsigned w, unsigned h) { m_width = w; m_height = h; return *this; }

                void serializeToSvgStream(std::ostream &stream) const;
            private:
                const RasterImage &m_image;
                Format m_format;
                unsigned m_jpgQuality;
                unsigned m_rescaleWidth, m_rescaleHeight;
                float m_x, m_y, m_width, m_height;

        };

        template<class FinalType>
        class Shape {
            public:
                Shape() { m_strokeWidth = 0.0f; }
                FinalType &setFill(const std::string &fill) { m_fill = fill; return (FinalType&)*this; }
                FinalType &setStroke(const std::string &stroke) { m_stroke = stroke; return (FinalType&)*this; }
                FinalType &setStrokeWidth(float width) { m_strokeWidth = width; return (FinalType&)*this; }
            protected:
                std::string m_fill;
                std::string m_stroke;
                float m_strokeWidth;
        };

        class Line : public Shape<Line> {
            public:
                Line(float x1, float y1, float x2, float y2);

                void serializeToSvgStream(std::ostream &stream) const;
            private:
                float m_x1, m_y1, m_x2, m_y2;
        };

        class Circle : public Shape<Circle> {
            public:
                Circle(float x, float y, float r);

                void serializeToSvgStream(std::ostream &stream) const;
            private:
                float m_x, m_y, m_radius;
        };

        class AARect : public Shape<AARect> {
            public:
                AARect(float x1, float y1, float x2, float y2);

                void serializeToSvgStream(std::ostream &stream) const;
            private:
                float m_x1, m_y1, m_x2, m_y2;
        };

        class MultiLine : public Shape<MultiLine> {
            public:
                void addSegment(float x1, float y1, float x2, float y2);

                void serializeToSvgStream(std::ostream &stream) const;
            private:
                struct LineSegment {
                    float x1, y1;
                    float x2, y2;
                };
                std::vector<LineSegment> m_segments;
        };


        template<class Type>
        SvgComposer& operator<<(const Type &element) {
            element.serializeToSvgStream(m_stream);
            return *this;
        }
    protected:
        std::ostream &m_stream;
};


/// @}

#endif // SVGCOMPOSER_H
