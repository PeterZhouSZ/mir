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

#include "CPUStopWatch.h"

#include <iostream>

namespace Engine {

CPUStopWatch::CPUStopWatch()
{
    start();
}


void CPUStopWatch::start()
{
    clock_gettime(CLOCK_MONOTONIC_RAW, &m_start);
}

uint64_t CPUStopWatch::getNanoseconds()
{
    timespec stop;
    clock_gettime(CLOCK_MONOTONIC_RAW, &stop);

    if (stop.tv_sec > m_start.tv_sec) {
        return (int64_t)(stop.tv_sec - m_start.tv_sec) * (int64_t)1000000000l
                    + ((int64_t) stop.tv_nsec - (int64_t)m_start.tv_nsec);
    } else {
        return stop.tv_nsec - m_start.tv_nsec;
    }
}

}
