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


#ifndef _HALFFLOAT_HPP_
#define _HALFFLOAT_HPP_

/**
 * @file
 * @author Andreas Ley
 */

/** @addtogroup Codebase_Group
 *  @{
 */

namespace HalfFloat
{
    union FloatInt {
        uint32_t i;
        float f;
    };


    inline float halfFloatToFloat(uint16_t hf)
    {
        unsigned sign = (hf >> 15) & 1;
        unsigned exponent = (hf >> 10) & ((1<<5)-1);//0b11111;
        unsigned mantissa = (hf >> 0) & ((1<<10)-1);//0b1111111111;

        mantissa = mantissa << (23-10);
        exponent = exponent + (127-15);

        FloatInt fi;
        fi.i = (sign << 31) | (exponent << 23) | (mantissa << 0);

        return fi.f;
    }

    inline uint16_t floatToHalfFloat(const float f)
    {
        FloatInt fi;
        fi.f = f;
        uint32_t floatData = fi.i;

        unsigned sign = (floatData >> 31) & 1;
        unsigned exponent = (floatData >> 23) & 0b11111111;
        unsigned mantissa = (floatData >> 0) & 0b11111111111111111111111;

        mantissa = mantissa >> (23-10);
        if (exponent >= (127-15))
            exponent = exponent - (127-15);
        else {
            unsigned remainder = (127-15) - exponent;
            exponent = 0;
            mantissa >>= remainder;
        }

        exponent &= 0b11111;
        mantissa &= 0b1111111111;

        uint16_t hf = (sign << 15) | (exponent << 10) | (mantissa << 0);

        return hf;
    }

}

/// @}

#endif // _HALFFLOAT_HPP_
