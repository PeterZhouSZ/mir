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

#include "Dataset.h"

#include <boost/lexical_cast.hpp>

#include <iostream>
#include <fstream>

namespace eyeTracking {

namespace {
std::istream& safeGetline(std::istream& is, std::string& t)
{
    t.clear();

    // The characters in the stream are read one-by-one using a std::streambuf.
    // That is faster than reading them one-by-one using the std::istream.
    // Code that uses streambuf this way must be guarded by a sentry object.
    // The sentry object performs various tasks,
    // such as thread synchronization and updating the stream state.

    std::istream::sentry se(is, true);
    std::streambuf* sb = is.rdbuf();

    for(;;) {
        int c = sb->sbumpc();
        switch (c) {
        case '\n':
            return is;
        case '\r':
            if(sb->sgetc() == '\n')
                sb->sbumpc();
            return is;
        case EOF:
            // Also handle the case when the last line has no line ending
            if(t.empty())
                is.setstate(std::ios::eofbit);
            return is;
        default:
            t += (char)c;
        }
    }
}
}

void Dataset::loadStimuli(const boost::filesystem::path &imgBasePath,
                    const std::string &categoriesFilename,
                    const std::string &imageCategoryLabelsFilename)
{
    m_imageBasePath = imgBasePath;
    m_images.clear();
    m_categories.clear();

	std::cout << "Reading categories from " << categoriesFilename << std::endl;
    {
        std::fstream categoriesFile;
        categoriesFile.open(categoriesFilename.c_str(), std::fstream::in);

        while (!categoriesFile.eof()) {
            std::string line;
            safeGetline(categoriesFile, line);
            if (line.empty()) {
                std::cout << "Warning: skipping empty line!" << std::endl;
                continue;
            }

            std::size_t comma = line.find(',');
            if (comma == std::string::npos)
                throw std::runtime_error("Missing comma in category file");

            unsigned categoryIndex = boost::lexical_cast<unsigned>(std::string(line.begin(), line.begin() + comma));
            if (categoryIndex != m_categories.size())
                throw std::runtime_error("Because andy is lazy, categories must be listed in ascending order without gaps!");

            std::string categoryName = std::string(line.begin() + comma+1, line.end());

            m_categories.push_back({categoryName});
        }
    }
	std::cout << "Reading image categories from " << imageCategoryLabelsFilename << std::endl;
    {
        std::fstream imagesFile;
        imagesFile.open(imageCategoryLabelsFilename.c_str(), std::fstream::in);

        while (!imagesFile.eof()) {
            std::string line;
            safeGetline(imagesFile, line);
            if (line.empty()) {
                std::cout << "Warning: skipping empty line!" << std::endl;
                continue;
            }

            std::size_t comma = line.find(',');
            if (comma == std::string::npos)
                throw std::runtime_error("Missing comma in labels file");

            Image image;

            image.filename = std::string(line.begin(), line.begin() + comma);


            image.category = boost::lexical_cast<unsigned>(std::string(line.begin() + comma+1, line.end()));
            if (image.category >= m_categories.size())
                throw std::runtime_error(std::string("invalid category index for file ") + image.filename);

            m_images.push_back(image);
        }
    }
}

unsigned Dataset::loadTestSubject(const boost::filesystem::path &path)
{


    boost::filesystem::path view = path / "viewing_samples.csv";
    boost::filesystem::path recall = path / "recall_samples.csv";

    TestSubject subject;
    if (boost::filesystem::exists(view))
        subject.loadViewingCSV(view.string());
    if (boost::filesystem::exists(recall))
        subject.loadRecallCSV(recall.string());

    subject.sanityCheckMaxImageIndex(m_images.size());

    subject.setName(path.filename().string());

    m_testSubjects.push_back(subject);
    return m_testSubjects.size()-1;
}

void Dataset::loadAllTestSubjects(const boost::filesystem::path &path)
{
    for (boost::filesystem::directory_iterator it(path); it != boost::filesystem::directory_iterator(); ++it) {
        if (boost::filesystem::is_directory(it->status())) {
            loadTestSubject(it->path());
        }
    }
}



}
