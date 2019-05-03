cmake_minimum_required (VERSION 2.6)

find_package (Boost COMPONENTS system filesystem unit_test_framework REQUIRED)

add_executable(unitTestConvNet
    source/testConvNetImplCudnn.cpp
    source/LayerTest.cpp
    source/LayerTest.h
)

target_include_directories (unitTestConvNet PUBLIC
    ${Boost_INCLUDE_DIRS}
)

add_definitions (-DBOOST_TEST_DYN_LINK)
target_link_libraries (unitTestConvNet
    convnet
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
)
