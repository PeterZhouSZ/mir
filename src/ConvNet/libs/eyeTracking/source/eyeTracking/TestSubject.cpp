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

#include "TestSubject.h"


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


void TestSubject::loadViewingCSV(const std::string &filename)
{
    m_viewingSequences.loadCSV(filename);
}

void TestSubject::loadRecallCSV(const std::string &filename)
{
    m_recallSequences.loadCSV(filename);
}

const EyeMovement *TestSubject::getViewingSequenceForImg(unsigned imgIdx) const
{
    return m_viewingSequences.getSequenceForImg(imgIdx);
}

const EyeMovement *TestSubject::getRecallSequenceForImg(unsigned imgIdx) const
{
    return m_recallSequences.getSequenceForImg(imgIdx);
}


void TestSubject::SequenceSet::loadCSV(const std::string &filename)
{
    std::fstream file;
    file.open(filename, std::fstream::in);

    unsigned currentImgIdx = ~0u;
    std::vector<EyeMovement::RawSample> rawSamples;

    auto flushRawSamples = [&]{
        if (!rawSamples.empty()) {
            m_sequences[currentImgIdx].setRawSamples(std::move(rawSamples));
            rawSamples.clear();
        }
    };

    while (!file.eof()) {
        std::string line;
        safeGetline(file, line);

        if (line.empty()) {
            std::cout << "Warning: skipping empty line!" << std::endl;
            continue;
        }

        if (std::string(line.begin(), line.begin()+5) == "Image") {
            flushRawSamples();
            currentImgIdx = boost::lexical_cast<unsigned>(std::string(line.begin()+6, line.end()));
        } else {
            if (currentImgIdx == ~0u)
                throw std::runtime_error("No image specified for eye sequence coordinates");

            std::size_t firstComma = line.find(',');
            if (firstComma == std::string::npos)
                throw std::runtime_error(std::string("Could not find comma in line ") + line);

            std::size_t secondComma = line.find(',', firstComma+1);
            if (secondComma == std::string::npos)
                throw std::runtime_error(std::string("Could not find second comma in line ") + line);

            EyeMovement::RawSample sample;
            sample.timestamp = boost::lexical_cast<double>(std::string(line.begin(), line.begin()+firstComma));
            sample.location[0] = boost::lexical_cast<double>(std::string(line.begin()+firstComma+1, line.begin()+secondComma));
            sample.location[1] = boost::lexical_cast<double>(std::string(line.begin()+secondComma+1, line.end()));
            rawSamples.push_back(sample);
        }
    }
    flushRawSamples();
}

const EyeMovement *TestSubject::SequenceSet::getSequenceForImg(unsigned imgIdx) const
{
    auto it = m_sequences.find(imgIdx);
    if (it == m_sequences.end())
        return nullptr;
    return &it->second;
}

void TestSubject::sanityCheckMaxImageIndex(unsigned numImages) const
{
    for (auto pair : m_viewingSequences.m_sequences)
        if (pair.first > numImages)
            throw std::runtime_error("Invalid image index found in viewing sequence!");

    for (auto pair : m_recallSequences.m_sequences)
        if (pair.first > numImages)
            throw std::runtime_error("Invalid image index found in recall sequence!");
}


}
