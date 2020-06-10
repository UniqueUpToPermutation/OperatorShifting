cmake_minimum_required(VERSION 3.7)
project(OperatorAugmentation)

include(CMakeModules/FindLEMON.cmake)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads)

set(CMAKE_CXX_STANDARD 14)

add_executable(OperatorAugmentation main.cpp augmentation.cpp dgnGridLaplacian1D.cpp
        dgnGridLaplacian2D.cpp diagnostics.cpp hfe.cpp meshlap.cpp dgnMeshLaplacian.cpp
        graphlap.cpp dgnGraphEdgePerturb.cpp matutil.cpp dgnGraphEdgeDrop.cpp
        auxAugmentation.cpp auxDiagnostics.h)
target_link_libraries(OperatorAugmentation Eigen3::Eigen Threads::Threads)