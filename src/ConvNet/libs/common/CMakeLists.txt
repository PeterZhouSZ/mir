cmake_minimum_required (VERSION 2.6)

find_package(ImageMagick COMPONENTS Magick++ REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread REQUIRED) 
    
add_library(common STATIC
    source/tools/CPUStopWatch.cpp
    source/tools/CPUStopWatch.h
    source/tools/HalfFloat.hpp
    source/tools/RasterImage.cpp
    source/tools/RasterImage.h
    source/tools/SvgComposer.h
    source/tools/SvgComposer.cpp
    source/tools/TaskScheduler.cpp
    source/tools/TaskScheduler.h
    source/tools/RefCountPtr.hpp
    source/tools/ChunkedArray.hpp
    source/tools/BinaryFStream.cpp
    source/tools/BinaryFStream.h
    source/tools/StackTrace.cpp
    source/tools/StackTrace.h
)



if (${ImageMagick_LIBRARIES} MATCHES "Q16")
target_compile_definitions(common PUBLIC "MAGICKCORE_QUANTUM_DEPTH=16")
else()
target_compile_definitions(common PUBLIC "MAGICKCORE_QUANTUM_DEPTH=8")
endif()

if (${ImageMagick_LIBRARIES} MATCHES "HDRI")
target_compile_definitions(common PUBLIC "MAGICKCORE_HDRI_ENABLE=true")
else()
target_compile_definitions(common PUBLIC "MAGICKCORE_HDRI_ENABLE=false")
endif()


target_include_directories(common PUBLIC
    ${ImageMagick_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)

target_include_directories(common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/source
)

target_link_libraries(common PUBLIC 
    ${ImageMagick_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    ${Boost_LIBRARIES}
)


#target_link_libraries(common PUBLIC
#    miscExtern
#)

