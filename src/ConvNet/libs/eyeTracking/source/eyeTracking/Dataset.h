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

#ifndef DATASET_H
#define DATASET_H

#include "TestSubject.h"


#include <boost/filesystem.hpp>

#include <string>
#include <vector>

namespace eyeTracking {

class Dataset
{
    public:
        struct Category {
            std::string name;
        };

        struct Image {
            std::string filename;
            unsigned category;
        };

        void loadStimuli(const boost::filesystem::path &imgBasePath,
                         const std::string &categoriesFilename,
                         const std::string &imageCategoryLabelsFilename);

        unsigned loadTestSubject(const boost::filesystem::path &path);

        void loadAllTestSubjects(const boost::filesystem::path &path);
        

        inline const boost::filesystem::path &getImageBasePath() const { return m_imageBasePath; }
        inline const std::vector<Image> &getImages() const { return m_images; }
        inline const std::vector<Category> &getCategories() const { return m_categories; }
        inline const std::vector<TestSubject> &getTestSubjects() const { return m_testSubjects; }
    protected:
        boost::filesystem::path m_imageBasePath;

        std::vector<Image> m_images;
        std::vector<Category> m_categories;

        std::vector<TestSubject> m_testSubjects;
};


}

#endif // DATASET_H
