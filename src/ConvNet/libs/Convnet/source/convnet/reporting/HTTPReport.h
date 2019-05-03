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


#pragma once
#ifndef _HTTPREPORT_H_
#define _HTTPREPORT_H_

#include <boost/filesystem.hpp>
#include <tools/RasterImage.h>


namespace convnet {

class HTTPReport
{
    public:
        HTTPReport(const boost::filesystem::path &targetDirectory,
                   const boost::filesystem::path &resourceDirectory,
                   bool softlink);

        void putImage(const RasterImage &image, const std::string &location);
        void putText(const std::stringstream &text, const std::string &location);
    protected:
        boost::filesystem::path m_targetDirectory;
        boost::filesystem::path m_resourceDirectory;
};

}

#endif // _HTTPREPORT_H_
