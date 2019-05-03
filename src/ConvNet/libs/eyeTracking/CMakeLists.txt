cmake_minimum_required (VERSION 2.6)

find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread filesystem REQUIRED)

    
add_library(eyeTracking STATIC
    source/eyeTracking/Dataset.h
    source/eyeTracking/Dataset.cpp
    source/eyeTracking/EyeMovement.h
    source/eyeTracking/EyeMovement.cpp
    source/eyeTracking/TestSubject.h
    source/eyeTracking/TestSubject.cpp
)



target_include_directories(eyeTracking PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)

target_include_directories(eyeTracking INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/source
)


target_link_libraries(eyeTracking PUBLIC
    common
    ${EIGEN3_LIBRARIES}
    ${Boost_LIBRARIES}
)

