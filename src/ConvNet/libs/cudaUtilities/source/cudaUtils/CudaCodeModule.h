/*
 * Cuda Utilities - Distributed for "Mental Image Retrieval" implementation
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

#ifndef CUDACODEMODULE_H
#define CUDACODEMODULE_H

#include <cuda.h>

namespace CudaUtils {

class CudaKernel;
class CudaTextureReference;
class CudaSurfaceReference;
class CudaConstantMemory;

class CudaCodeModule
{
    public:
        CudaCodeModule();
        ~CudaCodeModule();

        void loadFromFile(const char *filename);
        void loadFromMemory(const void *ptr);
        CudaKernel *getKernel(const char *kernelName);
/*
        CudaTextureReference *getTexReference(const char *name);
        CudaSurfaceReference *getSurfReference(const char *name);
        CudaConstantMemory *getConstantMemory(const char *name);
*/
    protected:
        bool m_handleValid;
        CUmodule m_moduleHandle;
};

}

#endif // CUDACODEMODULE_H
