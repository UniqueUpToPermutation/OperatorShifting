cmake_minimum_required(VERSION 3.16)
project(OperatorAugmentation)
#include(CMakeModules/FindSUPERLU.cmake)

#find_package(Armadillo REQUIRED)
#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

#include_directories(${ARMADILLO_INCLUDE_DIRS})
#include_directories(${SUPERLU_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD 14)

#add_compile_definitions(ARMA_USE_SUPERLU=1)
#add_compile_definitions(ARMA_SUPERLU_INCLUDE_DIR=${SUPERLU_INCLUDE_DIR}/)

add_executable(OperatorAugmentation main.cpp augmentation.cpp diagnostics.h dgnGridLaplacian1D.cpp dgnGridLaplacian2D.cpp)
target_link_libraries(OperatorAugmentation Eigen3::Eigen)