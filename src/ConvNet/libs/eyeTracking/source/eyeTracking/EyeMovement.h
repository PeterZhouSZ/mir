/*
 * Mental Image Retrieval - C++ Machine Learning implementation
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

#ifndef EYEMOVEMENT_H
#define EYEMOVEMENT_H

#include <Eigen/Dense>

#include <vector>

namespace eyeTracking {

class EyeMovement
{
    public:
        struct RawSample {
            std::size_t timestamp;
            Eigen::Vector2f location;
        };

        struct Fixation {
            std::size_t timestamp;
            float duration;
            Eigen::Vector2f location;
        };

        inline std::size_t getFirstTimestamp() const { return m_firstTimestamp; }
        inline const std::vector<RawSample> &getRawSamples() const { return m_rawSamples; }

        void setRawSamples(std::vector<RawSample> rawSamples);
    protected:
        std::size_t m_firstTimestamp;
        std::vector<RawSample> m_rawSamples;
        std::vector<Fixation> m_fixations;
};

}

#endif // EYEMOVEMENT_H
