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

#include "BinaryFStream.h"

namespace tools {

BinaryFStream::BinaryFStream(const char *filename, std::fstream::openmode openmode)
{ 
    m_stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);    
    m_stream.open(filename, openmode | std::fstream::binary);
}

void BinaryFStream::read(void *ptr, std::size_t size)
{
    m_stream.read((char*)ptr, size);
}

void BinaryFStream::write(const void *ptr, std::size_t size)
{
    m_stream.write((const char*)ptr, size);
}



}
