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


#ifndef TESTSUBJECT_H
#define TESTSUBJECT_H

#include "EyeMovement.h"

#include <map>

namespace eyeTracking {

class TestSubject
{
    public:

        void setName(std::string name) { m_name = std::move(name); }
        inline const std::string &getName() const { return m_name; }

        void loadViewingCSV(const std::string &filename);
        void loadRecallCSV(const std::string &filename);

        const EyeMovement *getViewingSequenceForImg(unsigned imgIdx) const;
        const EyeMovement *getRecallSequenceForImg(unsigned imgIdx) const;

        void sanityCheckMaxImageIndex(unsigned numImages) const;
    protected:
        std::string m_name;

        struct SequenceSet {
            std::map<unsigned, EyeMovement> m_sequences;

            void loadCSV(const std::string &filename);
            const EyeMovement *getSequenceForImg(unsigned imgIdx) const;
        };

        SequenceSet m_viewingSequences;
        SequenceSet m_recallSequences;
};

}

#endif // TESTSUBJECT_H
