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

#ifndef CUDNNERROR_H
#define CUDNNERROR_H

#include <exception>

#include <cudnn.h>

#include <string>


namespace cudnn {

class CuDnnError : public std::exception
{
    public:
        CuDnnError(cudnnStatus_t error, const char *invocation = nullptr, const char *file = nullptr, int line = -1);

        virtual const char* what() const noexcept;
        inline cudnnStatus_t getError() const { return m_error; }

        static std::string composeErrorMsg(cudnnStatus_t error, const char *invocation = nullptr, const char *file = nullptr, int line = -1);
    protected:
        std::string m_what;
        cudnnStatus_t m_error;
};

void checkCudnnErrorImpl(cudnnStatus_t error, const char *invocation = nullptr, const char *file = nullptr, int line = -1);
void throwCudnnErrorImpl(cudnnStatus_t error, const char *invocation = nullptr, const char *file = nullptr, int line = -1);

}

#define checkCudnnError(x) cudnn::checkCudnnErrorImpl(x, #x, __FILE__, __LINE__)
#define throwCudnnError(x) cudnn::throwCudnnErrorImpl(x, #x, __FILE__, __LINE__)

#endif // CUDNNERROR_H
