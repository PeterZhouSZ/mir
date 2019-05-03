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

#ifndef BLOBB_H
#define BLOBB_H

//#include <tools/CPUWarpReg.h>
#include <tools/RasterImage.h>
//#include <tools/AlignedAllocator.h>

#include <string.h>
#include <assert.h>

//using namespace CPUWarp;

#include "Tensor.h"

namespace convnet {

template<unsigned BatchSize>
class Blobb {
    public:
        void allocate(const TensorSize &dim) {
            m_dim = dim;

            m_xStride = BatchSize * m_dim.channels;
            m_yStride = BatchSize * m_dim.channels * m_dim.width;
            m_zStride = BatchSize * m_dim.channels * m_dim.width * m_dim.height;

            m_data.resize(m_dim.width*m_dim.height*m_dim.depth * m_dim.channels * BatchSize);
            memset(m_data.data(), 0xFF, m_data.size() * sizeof(float));
        }

        void setZero() {
//            for (float &f : m_data)
//                f = 0.0f;
            memset(m_data.data(), 0x00, m_data.size() * sizeof(float));
        }

        inline unsigned offsetOf(unsigned x, unsigned y, unsigned z, unsigned c, unsigned n) const {
            return n + c * BatchSize + x * m_xStride + y * m_yStride + z * m_zStride;
        }

        inline float &operator()(unsigned x, unsigned y, unsigned z, unsigned c, unsigned n) {
            assert(x < m_dim.width);
            assert(y < m_dim.height);
            assert(z < m_dim.depth);
            assert(c < m_dim.channels);
            assert(n < BatchSize);
            return m_data[offsetOf(x, y, z, c, n)];
        }
        inline const float &operator()(unsigned x, unsigned y, unsigned z, unsigned c, unsigned n) const  {
            assert(x < m_dim.width);
            assert(y < m_dim.height);
            assert(z < m_dim.depth);
            assert(c < m_dim.channels);
            assert(n < BatchSize);
            return m_data[offsetOf(x, y, z, c, n)];
        }

        inline float &operator[](unsigned i) {
            assert(i < m_data.size());
            return m_data[i];
        }
        inline const float &operator[](unsigned i) const  {
            assert(i < m_data.size());
            return m_data[i];
        }
/*
        inline CPUWarpRegFloat32<BatchSize> load(unsigned x, unsigned y, unsigned z, unsigned c) const {
            assert(x < m_dim.width);
            assert(y < m_dim.height);
            assert(z < m_dim.depth);
            assert(c < m_dim.channels);
            const float * __restrict__ ptr = m_data.data();
            return CPUWarpRegFloat32<BatchSize>::LoadSequentialAligned(ptr + offsetOf(x, y, z, c, 0));
        }

        template<_mm_hint locality>
        inline void prefetch(unsigned x, unsigned y, unsigned z, unsigned c) const {
            _mm_prefetch(m_data.data() + offsetOf(x, y, z, c, 0), locality);
        }

        inline void store(const CPUWarpRegFloat32<BatchSize> &data, unsigned x, unsigned y, unsigned z, unsigned c) {
            assert(x < m_dim.width);
            assert(y < m_dim.height);
            assert(z < m_dim.depth);
            assert(c < m_dim.channels);
            float * __restrict__ ptr = m_data.data();
            data.storeSequentialAligned(ptr + offsetOf(x, y, z, c, 0));
        }

        inline CPUWarpRegFloat32<BatchSize> load(unsigned i) const {
            assert(i < m_data.size());
            const float * __restrict__ ptr = m_data.data();
            return CPUWarpRegFloat32<BatchSize>::LoadSequentialAligned(ptr + i);
        }

        template<_mm_hint locality>
        inline void prefetch(unsigned i) const {
            _mm_prefetch(m_data.data() + i, locality);
        }

        inline void store(const CPUWarpRegFloat32<BatchSize> &data, unsigned i) {
            float * __restrict__ ptr = m_data.data();
            data.storeSequentialAligned(ptr + i);
        }
*/

        inline unsigned getWidth() const { return m_dim.width; }
        inline unsigned getHeight() const { return m_dim.height; }
        inline unsigned getDepth() const { return m_dim.depth; }
        inline unsigned getChannels() const { return m_dim.channels; }
        inline const TensorSize &getDimensions() const { return m_dim; }
        inline unsigned getBatchSize() const { return BatchSize; }
        inline unsigned getSize() const { return m_data.size(); }

	inline void negate();
	inline void setAll(float f);
        inline float sumSquareUpScalar() const;
        inline void operator*=(float scale);
        /*
        inline void negate() {
            for (unsigned i = 0; i < m_data.size(); i+= BatchSize) {
                CPUWarpRegFloat32<BatchSize> data = CPUWarpRegFloat32<BatchSize>::LoadSequentialAligned(m_data.data()+i);
                data = CPUWarp::negate(data);
                data.storeSequentialAligned(m_data.data()+i);
            }
        }

        inline void setAll(float f) {
            CPUWarpRegFloat32<BatchSize> data = f;
            for (unsigned i = 0; i < m_data.size(); i+= BatchSize) {
                data.storeSequentialAligned(m_data.data()+i);
            }
        }

        inline CPUWarpRegFloat32<BatchSize> sumUp() const {
            CPUWarpRegFloat32<BatchSize> sum;

            sum.setZero();
            for (unsigned i = 0; i < m_data.size(); i+= BatchSize) {
                CPUWarpRegFloat32<BatchSize> data = CPUWarpRegFloat32<BatchSize>::LoadSequentialAligned(m_data.data()+i);
                sum += data;
            }
            return sum;
        }

        inline CPUWarpRegFloat32<BatchSize> sumSquareUp() const {
            CPUWarpRegFloat32<BatchSize> sum;

            sum.setZero();
            for (unsigned i = 0; i < m_data.size(); i+= BatchSize) {
                CPUWarpRegFloat32<BatchSize> data = CPUWarpRegFloat32<BatchSize>::LoadSequentialAligned(m_data.data()+i);
                sum += data*data;
            }
            return sum;
        }

        inline float sumSquareUpScalar() const;

        inline CPUWarpRegFloat32<BatchSize> sumSquaredDistance(const Blobb<BatchSize> &other) const {
            CPUWarpRegFloat32<BatchSize> sum;

            sum.setZero();
            for (unsigned i = 0; i < m_data.size(); i+= BatchSize) {
                CPUWarpRegFloat32<BatchSize> data = CPUWarpRegFloat32<BatchSize>::LoadSequentialAligned(m_data.data()+i);
                CPUWarpRegFloat32<BatchSize> otherData = CPUWarpRegFloat32<BatchSize>::LoadSequentialAligned(other.m_data.data()+i);
                CPUWarpRegFloat32<BatchSize> diff = data - otherData;
                sum += diff*diff;
            }
            return sum;
        }

        inline void operator*=(float scale) {
            for (unsigned i = 0; i < m_data.size(); i+= BatchSize) {
                CPUWarpRegFloat32<BatchSize> data = CPUWarpRegFloat32<BatchSize>::LoadSequentialAligned(m_data.data()+i);
                data *= scale;
                data.storeSequentialAligned(m_data.data()+i);
            }
        }
*/
        Blobb<1> slice(unsigned n) const;

        inline unsigned totalSize() const { return m_data.size(); }
    protected:
        TensorSize m_dim;

        unsigned m_xStride = 0;
        unsigned m_yStride = 0;
        unsigned m_zStride = 0;

        //alignedVector<float> m_data;
        std::vector<float> m_data;
};


template<>
inline void Blobb<1>::negate() {
    for (unsigned i = 0; i < m_data.size(); i++) {
        m_data[i] = -m_data[i];
    }
}

template<>
inline void Blobb<1>::setAll(float f) {
    for (unsigned i = 0; i < m_data.size(); i++) {
        m_data[i] = f;
    }
}

template<>
inline float Blobb<1>::sumSquareUpScalar() const {
    float sum = 0.0f;

    for (unsigned i = 0; i < m_data.size(); i++) {
        float data = m_data[i];
        sum += data*data;
    }
    return sum;
}

template<>
inline void Blobb<1>::operator*=(float scale) {
    for (unsigned i = 0; i < m_data.size(); i++) {
        m_data[i] *= scale;
    }
}



template<unsigned BatchSize>
Blobb<1> Blobb<BatchSize>::slice(unsigned n) const
{
    Blobb<1> result;
    result.allocate(m_dim);
    for (unsigned i = 0; i < result.getSize(); i++)
        result[i] = m_data[i*BatchSize + n];
    return result;
}


template<unsigned BatchSize, bool singleChannel>
std::vector<Blobb<BatchSize>> splitFile(const char *filename, unsigned w, unsigned h)
{
    RasterImage img;
    img.loadFromFile(filename);

    unsigned cropCols = img.getWidth() / w;
    unsigned cropRows = img.getHeight() / h;
    unsigned numTotalCrops = cropCols * cropRows;
    unsigned numBlobbs = numTotalCrops / BatchSize;

    std::vector<Blobb<BatchSize>> result;
    result.resize(numBlobbs);
    for (auto &b : result)
        b.allocate(TensorSize(w, h, 1, singleChannel?1:3));

    for (unsigned r = 0; r < cropRows; r++)
        for (unsigned c = 0; c < cropCols; c++) {
            unsigned cropIdx = r*cropCols + c;
            unsigned blobbIdx = cropIdx / BatchSize;
            unsigned imgIdx = cropIdx % BatchSize;
            if (blobbIdx >= numBlobbs) break;

            for (unsigned y = 0; y < h; y++)
                for (unsigned x = 0; x < w; x++) {
                    if (singleChannel) {
                        int v = img.getData()[(x+c*w) + (y+r*h)*img.getWidth()] & 0xFF;
                        result[blobbIdx](x, y, 0, 0, imgIdx) = v / 255.0f * 2.0f - 1.0f;
                    } else {
                        int v = img.getData()[(x+c*w) + (y+r*h)*img.getWidth()];
                        int r = (v >> 0) & 0xFF;
                        int g = (v >> 8) & 0xFF;
                        int b = (v >> 16) & 0xFF;
                        result[blobbIdx](x, y, 0, 0, imgIdx) = r / 255.0f * 2.0f - 1.0f;
                        result[blobbIdx](x, y, 0, 1, imgIdx) = g / 255.0f * 2.0f - 1.0f;
                        result[blobbIdx](x, y, 0, 2, imgIdx) = b / 255.0f * 2.0f - 1.0f;
                    }
                }
        }

    return result;
}


template<unsigned BatchSize>
void writeToFile(const char *filename, unsigned z, unsigned c, unsigned n, const Blobb<BatchSize> &blobb, float scale = 1.0f, float offset = 0.0f)
{
    RasterImage img;
    img.resize(blobb.getWidth(), blobb.getHeight());
    for (unsigned y = 0; y < blobb.getHeight(); y++) {
        for (unsigned x = 0; x < blobb.getWidth(); x++) {
            float v = blobb(x, y, z, c, n) * scale + offset;
            int iv = std::min<int>(std::max<int>(v, 0), 255);
            img.getData()[x+y*blobb.getWidth()] =
                            (iv << 0) |
                            (iv << 8) |
                            (iv << 16) |
                            (0xFF << 24);
        }
    }
    img.writeToFile(filename);
}

template<unsigned BatchSize>
void writeToFile(const char *filename, unsigned z, unsigned n, const Blobb<BatchSize> &blobb, float scale = 1.0f, float offset = 0.0f)
{
    if (blobb.getChannels() != 3) {
        writeToFile<BatchSize>(filename, z, 0, n, blobb, scale, offset);
    } else {
        RasterImage img;
        img.resize(blobb.getWidth(), blobb.getHeight());
        for (unsigned y = 0; y < blobb.getHeight(); y++) {
            for (unsigned x = 0; x < blobb.getWidth(); x++) {
                float r = blobb(x, y, z, 0, n) * scale + offset;
                float g = blobb(x, y, z, 1, n) * scale + offset;
                float b = blobb(x, y, z, 2, n) * scale + offset;
                int ir = std::min<int>(std::max<int>(r, 0), 255);
                int ig = std::min<int>(std::max<int>(g, 0), 255);
                int ib = std::min<int>(std::max<int>(b, 0), 255);
                img.getData()[x+y*blobb.getWidth()] =
                                (ir << 0) |
                                (ig << 8) |
                                (ib << 16) |
                                (0xFF << 24);
            }
        }
        img.writeToFile(filename);
    }
}

template<unsigned BatchSize>
void writeChannelsToFile(const char *filename, unsigned z, unsigned c_r, unsigned c_g, unsigned c_b, unsigned n, const Blobb<BatchSize> &blobb, float scale = 1.0f, float offset = 0.0f)
{
    RasterImage img;
    img.resize(blobb.getWidth(), blobb.getHeight());
    for (unsigned y = 0; y < blobb.getHeight(); y++) {
        for (unsigned x = 0; x < blobb.getWidth(); x++) {
            float r = blobb(x, y, z, c_r, n) * scale + offset;
            float g = blobb(x, y, z, c_g, n) * scale + offset;
            float b = blobb(x, y, z, c_b, n) * scale + offset;
            int ir = std::min<int>(std::max<int>(r, 0), 255);
            int ig = std::min<int>(std::max<int>(g, 0), 255);
            int ib = std::min<int>(std::max<int>(b, 0), 255);
            img.getData()[x+y*blobb.getWidth()] =
                            (ir << 0) |
                            (ig << 8) |
                            (ib << 16) |
                            (0xFF << 24);
        }
    }
    img.writeToFile(filename);
}

/*
template<unsigned BatchSize>
float computeAvgSquaredDistance(const Blobb<BatchSize> &a, const Blobb<BatchSize> &b)
{
    CPUWarpRegFloat32<BatchSize> sum;
    sum.setZero();
    for (unsigned z = 0; z < a.getDepth(); z++) {
        for (unsigned y = 0; y < a.getHeight(); y++) {
            for (unsigned x = 0; x < a.getWidth(); x++) {
                for (unsigned c = 0; c < a.getChannels(); c++) {
                    CPUWarpRegFloat32<BatchSize> av = a.load(x, y, z, c);
                    CPUWarpRegFloat32<BatchSize> bv = b.load(x, y, z, c);
                    CPUWarpRegFloat32<BatchSize> d = av-bv;
                    sum += d*d;
                }
            }
        }
    }
    float res;
    sum.storeHorizontalSum(&res);
    return res / (BatchSize * a.getWidth() * a.getHeight() * a.getDepth() * a.getChannels());
}
*/

}

#endif // BLOBB_H
