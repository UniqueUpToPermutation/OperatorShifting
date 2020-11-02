cmake_minimum_required(VERSION 3.7)
project(OperatorAugmentation)

include(CMakeModules/FindLEMON.cmake)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads)

set(CMAKE_CXX_STANDARD 14)

add_executable(OperatorAugmentation main.cpp augmentation.cpp dgnGridLaplacian1D.cpp
        dgnGridLaplacian2D.cpp diagnostics.cpp
        graphlap.cpp dgnGraphEdgeDrop.cpp matutil.cpp dgnGraphEdgePerturb.cpp
        auxAugmentation.cpp auxDiagnostics.h testbeds.h)

target_link_libraries(OperatorAugmentation Eigen3::Eigen Threads::Threads)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config_default.json
        ${CMAKE_CURRENT_BINARY_DIR}/config_default.json COPYONLY)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Graphs DESTINATION ${CMAKE_CURRENT_BINARY_DIR})