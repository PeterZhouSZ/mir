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

#ifndef BINARYFSTREAM_H
#define BINARYFSTREAM_H

#include <vector>
#include <string>
#include <cstdint>
#include <type_traits>
#include <fstream>

namespace tools {

struct BinaryFStream
{
    public:
        BinaryFStream(const char *filename, std::fstream::openmode openmode);

        template<typename type, typename castType>
        BinaryFStream &readAs(type &var) {
            castType tmp;
            (*this)>>tmp;
            var = (type) tmp;
            return *this;
        }
        template<typename type, typename castType>
        BinaryFStream &writeAs(type &var) {
            castType tmp = (castType) var;
            (*this)<<tmp;
            return *this;
        }

        void read(void *ptr, std::size_t size);
        void write(const void *ptr, std::size_t size);

        template<typename Type,
                 typename T = typename std::enable_if<std::is_trivially_copyable<Type>::value>::type>
        inline BinaryFStream &operator<<(const Type &var) {
            static_assert(!std::is_pointer<Type>::value, "Writing pointers to file not a good idea!");
            write(&var, sizeof(Type));
            return *this;
        }

        template<typename Type,
                 typename T = typename std::enable_if<std::is_trivially_copyable<Type>::value>::type>
        inline BinaryFStream &operator>>(Type &var) {
            static_assert(!std::is_pointer<Type>::value, "Reading pointers from file not a good idea!");
            read(&var, sizeof(Type));
            return *this;
        }


        inline BinaryFStream &operator<<(const std::string &var) {
            std::uint16_t len = var.length();
            write(&len, sizeof(len));
            write(var.data(), len);
            return *this;
        }
        inline BinaryFStream &operator>>(std::string &var) {
            std::uint16_t len;
            read(&len, sizeof(len));
            var.resize(len);
            //read(var.data(), len);
            read(&var[0], len);
            return *this;
        }

        
        
        
        template<typename Type>
        inline BinaryFStream &operator<<(const std::vector<typename std::enable_if<std::is_trivially_copyable<Type>::value, Type>::type> &var) {
            static_assert(!std::is_pointer<Type>::value, "Writing pointers to file not a good idea!");
            std::uint32_t len = var.size();
            write(&len, sizeof(len));
            write(var.data(), len*sizeof(Type));
            return *this;
        }

        template<typename Type>
        inline BinaryFStream &operator>>(std::vector<typename std::enable_if<std::is_trivially_copyable<Type>::value, Type>::type> &var) {
            static_assert(!std::is_pointer<Type>::value, "Reading pointers from file not a good idea!");
            std::uint32_t len;
            read(&len, sizeof(len));
            var.resize(len);
            read(var.data(), len*sizeof(Type));
            return *this;
        }

        template<typename Type>
        inline BinaryFStream &operator<<(const std::vector<Type> &var) {
            std::uint32_t len = var.size();
            write(&len, sizeof(len));
            for (unsigned i = 0; i < len; i++)
                (*this) << var[i];
            return *this;
        }

        template<typename Type>
        inline BinaryFStream &operator>>(std::vector<Type> &var) {
            std::uint32_t len;
            read(&len, sizeof(len));
            var.resize(len);
            for (unsigned i = 0; i < len; i++)
                (*this) >> var[i];
            return *this;
        }    
                

    private:
        std::fstream m_stream;
};

}

#endif // BINARYFSTREAM_H
