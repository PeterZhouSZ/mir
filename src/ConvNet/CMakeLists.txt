cmake_minimum_required (VERSION 2.8)
project (Everything)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

LIST(APPEND CMAKE_CXX_FLAGS "-march=native -ffast-math -Wall -pedantic -Wextra -std=c++17")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -O3 -g -DNDEBUG")


option(USE_ISPC "Compile with ispc" FALSE)

if(USE_ISPC)
IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(ispc_compiler $ENV{ISPC_HOME}/ispc --opt=fast-math --opt=force-aligned-memory --opt=fast-masked-vload --arch=x86_64 --cpu=core-avx2 -O0 -g)
ELSE()
    set(ispc_compiler $ENV{ISPC_HOME}/ispc --opt=fast-math --opt=force-aligned-memory --opt=fast-masked-vload --opt=disable-assertions --arch=x86_64 --cpu=core-avx2 -O3 -g)
ENDIF()
endif(USE_ISPC)


#add_subdirectory(external/misc)
#add_subdirectory(external/cnpy/)
add_subdirectory(libs/common)
add_subdirectory(libs/cudaUtilities/)
add_subdirectory(libs/Convnet)
add_subdirectory(libs/eyeTracking)
add_subdirectory(executables/AndysSandbox)

