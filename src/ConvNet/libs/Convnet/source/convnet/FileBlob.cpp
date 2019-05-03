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

#include "FileBlob.h"


namespace convnet {

void FileBlob::openForReading(const std::string &filename)
{
    m_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    m_file.open(filename.c_str(), std::fstream::in | std::fstream::binary);
}

void FileBlob::openForWriting(const std::string &filename)
{
    m_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    m_file.open(filename.c_str(), std::fstream::out | std::fstream::binary);
}

void FileBlob::read(void *ptr, std::size_t size)
{
    m_file.read((char*)ptr, size);
}

void FileBlob::write(const void *ptr, std::size_t size)
{
    m_file.write((const char*)ptr, size);
}


}
