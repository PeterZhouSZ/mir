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

#include "FilterRenderer.h"

#include "layers/ConvolutionLayer.h"
//#include "layers/TransposedConvolutionLayer.h"

#include <fstream>

namespace convnet {

inline void YCbCr2RGB(float Y, float Cb, float Cr, float &r, float &g, float &b)
{
    r = 1.0f * Y + 0.0f * Cb + 1.402f * Cr;
    g = 1.0f * Y + -0.344f * Cb + -0.71414 * Cr;
    b = 1.0f * Y + 1.77f * Cb + 0.0f * Cr;
}

template<class FilterClass>
void FilterRenderer::render(const FilterClass &convParams,
            const unsigned rgbChannels[3],
            unsigned depthSlice,
            float offset, float scale,
            const std::string &filename,
            ColorSpaceConversion colorConv)
{


    unsigned cols = std::ceil(std::sqrt(convParams.m_kernel.size()));
    unsigned rows = (convParams.m_kernel.size() + cols-1) / cols;

    unsigned filterW = convParams.m_filterSize.width;
    unsigned filterH = convParams.m_filterSize.height;

    unsigned width = cols * filterW * m_res + (cols-1) * m_innerPadding + 2*m_outerPadding;
    unsigned height = rows * filterH * m_res + (rows-1) * m_innerPadding + 2*m_outerPadding;

    std::fstream svgFile(filename.c_str(), std::fstream::out);
    svgFile
        << "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>" << std::endl
        << "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\""<<width<<"\" height=\""<<height<<"\">" << std::endl
        << "    <rect x=\"0\" y=\"0\" width=\""<<width<<"\" height=\""<<height<<"\" fill=\"white\" stroke=\"black\"/>" << std::endl;

    for (unsigned filterIdx = 0; filterIdx < convParams.m_kernel.size(); filterIdx++) {
        unsigned c = filterIdx % cols;
        unsigned r = filterIdx / cols;

        unsigned offsetX = m_outerPadding + c * filterW * m_res + c * m_innerPadding;
        unsigned offsetY = m_outerPadding + r * filterH * m_res + r * m_innerPadding;

        for (unsigned y = 0; y < filterH; y++) {
            for (unsigned x = 0; x < filterW; x++) {
                unsigned x0 = offsetX + x * m_res;
                unsigned y0 = offsetY + y * m_res;

                float fr = rgbChannels[0]!=~0u?convParams.m_kernel[filterIdx].filter(x, y, depthSlice, rgbChannels[0], 0):0.0f;
                float fg = rgbChannels[1]!=~0u?convParams.m_kernel[filterIdx].filter(x, y, depthSlice, rgbChannels[1], 0):0.0f;
                float fb = rgbChannels[2]!=~0u?convParams.m_kernel[filterIdx].filter(x, y, depthSlice, rgbChannels[2], 0):0.0f;

                switch (colorConv) {
                    case ColorConv_None:
                    break;
                    case ColorConv_YCbCr2RGB:
                        YCbCr2RGB(fr, fg, fb, fr, fg, fb);
                    break;
                }

                int r = rgbChannels[0]!=~0u?std::min<int>(std::max<int>(fr * scale + offset, 0), 255):0;
                int g = rgbChannels[1]!=~0u?std::min<int>(std::max<int>(fg * scale + offset, 0), 255):0;
                int b = rgbChannels[2]!=~0u?std::min<int>(std::max<int>(fb * scale + offset, 0), 255):0;

                svgFile
                    << "    <rect x=\""<<x0<<"\" y=\""<<y0<<"\" width=\""<<m_res<<"\" height=\""<<m_res<<"\" fill=\"rgb("<<r<<','<<g<<','<<b<<")\" stroke=\"none\"/>"<<std::endl;
            }
        }
    }

    svgFile << "</svg>" << std::endl;
}



void FilterRenderer::render(const ConvolutionParameters &convParams,
            const unsigned rgbChannels[3],
            unsigned depthSlice,
            float offset, float scale,
            const std::string &filename,
            ColorSpaceConversion colorConv)
{
    render<ConvolutionParameters>(convParams, rgbChannels, depthSlice, offset, scale, filename, colorConv);
}
/*
void FilterRenderer::render(const TransposedConvolutionParameters &convParams,
            const unsigned rgbChannels[3],
            unsigned depthSlice,
            float offset, float scale,
            const std::string &filename,
            ColorSpaceConversion colorConv)
{
    render<TransposedConvolutionParameters>(convParams, rgbChannels, depthSlice, offset, scale, filename, colorConv);
}
*/

}
