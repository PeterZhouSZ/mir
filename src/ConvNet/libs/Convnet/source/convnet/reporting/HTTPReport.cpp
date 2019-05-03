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

#include "HTTPReport.h"

#include <fstream>

namespace convnet {
    
HTTPReport::HTTPReport(const boost::filesystem::path &targetDirectory,
                       const boost::filesystem::path &resourceDirectory,
                       bool softlink) : m_targetDirectory(targetDirectory), m_resourceDirectory(resourceDirectory)
{
    boost::filesystem::create_directories(m_targetDirectory);
    
    auto handleDir = [&](const char *subdir) {
        boost::filesystem::create_directories(m_targetDirectory / subdir);
        for (boost::filesystem::directory_iterator it(m_resourceDirectory / subdir); it != boost::filesystem::directory_iterator(); ++it) {
            if (boost::filesystem::is_regular_file(it->path())) {    
                boost::filesystem::copy_file(
                    it->path(),
                    m_targetDirectory / subdir / it->path().filename(),
                    boost::filesystem::copy_option::overwrite_if_exists
                );
            }
        }
    };
    handleDir("js");
    handleDir("css");
    handleDir("fonts");
}

void atomicRename(const boost::filesystem::path &src, const boost::filesystem::path &dst) 
{
    rename(src.string().c_str(), dst.string().c_str());
}

void HTTPReport::putImage(const RasterImage &image, const std::string &location)
{
    boost::filesystem::path target = m_targetDirectory / location;
    boost::filesystem::create_directories(target.parent_path());
    boost::filesystem::path tmpPath = target;
    tmpPath.replace_extension(std::string("_tmp") + target.extension().string());
    image.writeToFile(tmpPath.string().c_str());
    atomicRename(tmpPath, target);
}

void HTTPReport::putText(const std::stringstream &text, const std::string &location)
{
    boost::filesystem::create_directories((m_targetDirectory / location).parent_path());
    std::fstream file;
    file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    file.open((m_targetDirectory / (location + "_tmp")).string().c_str(), std::fstream::out);
    file << text.str();
    atomicRename(m_targetDirectory / (location + "_tmp"), m_targetDirectory / location);
}

}
