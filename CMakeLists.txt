cmake_minimum_required(VERSION 3.7)
project(OperatorAugmentation)

include(CMakeModules/FindLEMON.cmake)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads)

set(CMAKE_CXX_STANDARD 17)

set(OPSHIFT_SRC_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(OPSHIFT_SRC
        ${OPSHIFT_SRC_DIR}/diagnostics.cpp
        ${OPSHIFT_SRC_DIR}/graphlap.cpp
        ${OPSHIFT_SRC_DIR}/matutil.cpp
        ${OPSHIFT_SRC_DIR}/opshift.cpp
)

set(OPSHIFT_INCLUDE_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(OPSHIFT_LIBRARIES Eigen3::Eigen Threads::Threads)

set(OPSHIFT_TEST_OUTPUT_DIR
        ${CMAKE_CURRENT_BINARY_DIR})

function(CreateTest TARGET_NAME TEST_SRC)
        add_executable(${TARGET_NAME} ${OPSHIFT_SRC} ${TEST_SRC})
        target_include_directories(${TARGET_NAME} PUBLIC ${OPSHIFT_INCLUDE_DIR})
        target_link_libraries(${TARGET_NAME} PUBLIC ${OPSHIFT_LIBRARIES})
        SET_TARGET_PROPERTIES(${TARGET_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${OPSHIFT_TEST_OUTPUT_DIR}
        )
endfunction()

add_subdirectory(tests)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config_default.json
        ${CMAKE_CURRENT_BINARY_DIR}/config_default.json COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.json
        ${CMAKE_CURRENT_BINARY_DIR}/config.json COPYONLY)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Graphs DESTINATION ${CMAKE_CURRENT_BINARY_DIR})