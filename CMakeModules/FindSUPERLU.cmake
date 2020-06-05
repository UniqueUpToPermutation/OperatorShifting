# - Find the SUPERLU includes and library
#
# This module defines
#  SUPERLU_INCLUDE_DIR, where to find supermatrix.h, etc.
#  SUPERLU_LIBRARIES, the libraries to link against to use SUPERLU.
#  SUPERLU_FOUND, If false, do not try to use SUPERLU.
# also defined, but not for general use are

#=============================================================================
# Copyright 2011, Martin Koehler
# http://www-user.tu-chemnitz.de/~komart/
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# Changelog:
#    - Nov. 3, 2015 	Maximilian Behr

if (NOT _incdir)
    if (WIN32)
        set(_incdir ENV INCLUDE)
    elseif (APPLE)
        set(_incdir ENV INCLUDE CPATH)
    else ()
        set(_incdir ENV INCLUDE CPATH)
    endif ()
endif ()

if (NOT _libdir)
    if (WIN32)
        set(_libdir ENV LIB)
    elseif (APPLE)
        set(_libdir ENV DYLD_LIBRARY_PATH)
    else ()
        set(_libdir ENV LD_LIBRARY_PATH)
    endif ()
endif ()

#look for header in user given directories
find_path(SUPERLU_SUPERLU_INCLUDE_DIR
        NAMES supermatrix.h
        PATHS ${SUPERLU_ROOT}
        PATH_SUFFIXES "superlu" "SuperLU" "include/superlu" "include" "SRC"
        NO_DEFAULT_PATH
        )

#look for header in standard paths
find_path(SUPERLU_SUPERLU_INCLUDE_DIR
        NAMES supermatrix.h
        PATHS ${_incdir} /usr/include /usr/local/include /opt/local/include
        PATH_SUFFIXES "superlu" "SuperLU"
        )

#find library
set(SUPERLU_NAMES "superlu" "superlu_2.0" "superlu_3.0" "superlu_3.1" "superlu_4.0" "superlu_4.1" "superlu_4.2" "superlu_4.3" "superlu_5.0"
        "superlu_mt" "superlu_mt_2.0" "superlu_mt_2.1" "superlu_mt_2.2" "superlu_mt_2.3" "superlu_mt_2.4" "superlu_mt_3.0")

#look for libraries in user given path
find_library(SUPERLU_LIBRARY
        NAMES ${SUPERLU_NAMES}
        PATHS ${SUPERLU_ROOT}
        PATH_SUFFIXES "lib" "lib32" "lib64"
        NO_DEFAULT_PATH
        )


#look for libraries in standard path
find_library(SUPERLU_LIBRARY
        NAMES ${SUPERLU_NAMES}
        PATHS ${_libdir} /usr/lib /usr/local/lib /opt/local/lib
        PATH_SUFFIXES "superlu" "SuperLU"
        )


SET(SUPERLU_FOUND FALSE)
IF (SUPERLU_LIBRARY AND SUPERLU_SUPERLU_INCLUDE_DIR)
    SET(SUPERLU_FOUND TRUE)
    SET(SUPERLU_INCLUDE_DIR ${SUPERLU_SUPERLU_INCLUDE_DIR})

    #set includes and libraries for tests
    #SET(CMAKE_REQUIRED_LIBRARIES "${SUPERLU_LIBRARY}" "${BLAS_LIBRARIES}" "-lm")
    SET(CMAKE_REQUIRED_LIBRARIES "${SUPERLU_LIBRARY}" "${BLAS_LIBRARIES}" "-lm")
    SET(CMAKE_REQUIRED_INCLUDES ${SUPERLU_SUPERLU_INCLUDE_DIR})
    MESSAGE("Found SuperLU!")
ENDIF()



# handle the QUIETLY and REQUIRED arguments and set SUPERLU_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SUPERLU DEFAULT_MSG SUPERLU_LIBRARY SUPERLU_SUPERLU_INCLUDE_DIR )

mark_as_advanced(SUPERLU SUPERLU_SUPERLU_INCLUDE_DIR )
