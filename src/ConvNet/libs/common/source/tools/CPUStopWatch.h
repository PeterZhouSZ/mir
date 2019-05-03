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

#ifndef CPUSTOPWATCH_H
#define CPUSTOPWATCH_H

#include <time.h>
#include <stdint.h>

/** @addtogroup Codebase_Group
 *  @{
 */

namespace Engine {

class CPUStopWatch
{
    public:
        CPUStopWatch();

        void start();
        uint64_t getNanoseconds();
    protected:
        timespec m_start;
};

}

/// @}

#endif // CPUSTOPWATCH_H
