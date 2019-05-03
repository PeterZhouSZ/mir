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

#ifndef _REFCOUNTPTR_HPP_
#define _REFCOUNTPTR_HPP_


template<class HostType>
class RefCountPtr {
    public:
        RefCountPtr() {
            m_ptr = nullptr;
        }
        RefCountPtr(const RefCountPtr<HostType> &other) {
            m_ptr = other.m_ptr;
            if (m_ptr != nullptr)
                m_ptr->incReferences();
        }
        RefCountPtr(HostType *other) {
            m_ptr = other;
            if (m_ptr != nullptr)
                m_ptr->incReferences();
        }
        ~RefCountPtr() {
            if (m_ptr != nullptr)
                m_ptr->decReferences();
        }
        const RefCountPtr<HostType> &operator=(const RefCountPtr<HostType> &other) {
            if (&other == this)
                return *this;

            if (m_ptr != nullptr)
                m_ptr->decReferences();

            m_ptr = other.m_ptr;
            if (m_ptr != nullptr)
                m_ptr->incReferences();
            return *this;
        }
        const RefCountPtr<HostType> &operator=(HostType *other) {
            if (m_ptr != nullptr)
                m_ptr->decReferences();

            m_ptr = other;
            if (m_ptr != nullptr)
                m_ptr->incReferences();
            return *this;
        }

        inline HostType *operator->() { return m_ptr; }
        inline const HostType *operator->() const { return m_ptr; }

        inline operator HostType*() { return m_ptr; }
        inline HostType *get() { return m_ptr; }
        inline const HostType *get() const { return m_ptr; }

        bool operator==(const RefCountPtr<HostType> &other) const { return m_ptr == other.m_ptr; }
    private:
        HostType *m_ptr;
};


#endif // _REFCOUNTPTR_HPP_
