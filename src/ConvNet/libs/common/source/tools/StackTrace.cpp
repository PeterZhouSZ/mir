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

#include "StackTrace.h"

#include <execinfo.h>

#include <iostream>
#include <iomanip>

StackTrace::~StackTrace()
{
    if (m_symbols != nullptr)
        free(m_symbols);
}

void StackTrace::capture()
{
    m_traceSize = backtrace(m_trace, 50);    
    if (m_symbols != nullptr)
        free(m_symbols);
    m_symbols = backtrace_symbols(m_trace, m_traceSize);
}

void StackTrace::print()
{
    for (unsigned i = 0; i < m_traceSize; i++) {
        std::cout << std::setw(2) << i << "  " << std::setw(16) << std::hex << m_trace[i] << std::dec << "  " << m_symbols[i] << std::endl;
        std::cout << std::setw(1);
    }
}

