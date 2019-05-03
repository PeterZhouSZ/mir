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

#include "SvgComposer.h"
#include <Magick++.h>


SvgComposer::SvgComposer(std::ostream &stream, unsigned width, unsigned height) : m_stream(stream)
{
    m_stream << "<?xml version=\"1.0\" standalone=\"no\"?>" << std::endl;
    m_stream << "<svg width=\""<<width<<"\" height=\""<<height<<"\" version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\">" << std::endl;
}

SvgComposer::~SvgComposer()
{
    m_stream << "</svg>" << std::endl;
}

SvgComposer::Image::Image(const RasterImage &image, float x, float y) : m_image(image)
{
    m_format = FORMAT_JPG;
    m_jpgQuality = 90;
    m_rescaleWidth = 0;
    m_rescaleHeight = 0;

    m_x = x;
    m_y = y;
    m_width = m_image.getWidth();
    m_height = m_image.getHeight();
}


void SvgComposer::Image::serializeToSvgStream(std::ostream &stream) const
{
    Magick::Image image(m_image.getWidth(), m_image.getHeight(), "RGBA", Magick::CharPixel, m_image.getData());
    if ((m_rescaleWidth != 0) && (m_rescaleHeight != 0))
        image.resize(Magick::Geometry(m_rescaleWidth, m_rescaleHeight));
    image.quality(m_jpgQuality);

    stream << "    <image x=\""<< m_x <<"\" y=\""<< m_y << "\" width=\""<<m_width<<"\" height=\""<<m_height<<"\" ";

    switch (m_format) {
        case FORMAT_JPG:
            image.magick("JPEG");
            stream << "xlink:href=\"data:image/jpeg;base64,";
        break;
        case FORMAT_PNG:
            image.magick("PNG");
            stream << "xlink:href=\"data:image/png;base64,";
        break;
    }
    Magick::Blob blob;
    image.write(&blob);
    stream << blob.base64();
    stream << "\" />" << std::endl;

}


SvgComposer::Line::Line(float x1, float y1, float x2, float y2)
{
    m_x1 = x1;
    m_y1 = y1;
    m_x2 = x2;
    m_y2 = y2;
}

void SvgComposer::Line::serializeToSvgStream(std::ostream &stream) const
{
    stream << "<line x1=\"" << m_x1 << "\" y1=\"" << m_y1 << "\" x2=\"" << m_x2 << "\" y2=\"" << m_y2 << "\""
            << " stroke=\"" << m_stroke << "\" stroke-width=\"" << m_strokeWidth << "\" />" << std::endl;
}


SvgComposer::Circle::Circle(float x, float y, float r)
{
    m_x = x;
    m_y = y;
    m_radius = r;
}

void SvgComposer::Circle::serializeToSvgStream(std::ostream &stream) const
{
    stream << "<circle cx=\"" << m_x << "\" cy=\"" << m_y << "\" r=\"" << m_radius << "\""
            << " fill=\"" << m_fill << "\" stroke=\"" << m_stroke << "\" stroke-width=\"" << m_strokeWidth << "\" />" << std::endl;
}

SvgComposer::AARect::AARect(float x1, float y1, float x2, float y2)
{
    m_x1 = x1;
    m_y1 = y1;
    m_x2 = x2;
    m_y2 = y2;
}

void SvgComposer::AARect::serializeToSvgStream(std::ostream &stream) const
{
    stream << "<rect x=\"" << m_x1 << "\" y=\"" << m_y1 << "\" width=\"" << m_x2-m_x1 << "\" height=\"" << m_y2-m_y1 << "\""
            << " stroke=\"" << m_stroke << "\" stroke-width=\"" << m_strokeWidth << "\" />" << std::endl;
}

void SvgComposer::MultiLine::addSegment(float x1, float y1, float x2, float y2)
{
    m_segments.push_back({x1, y1, x2, y2});
}

void SvgComposer::MultiLine::serializeToSvgStream(std::ostream &stream) const
{
    stream << "<path d=\"";
    for (const LineSegment &segment : m_segments)
        stream << " M " << segment.x1<<','<< segment.y1 << ' ' << segment.x2<<',' << segment.y2;
    stream << "\" stroke=\"" << m_stroke << "\" stroke-width=\"" << m_strokeWidth << "\" />" << std::endl;
}


