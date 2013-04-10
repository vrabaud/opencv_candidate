cmake_minimum_required(VERSION 2.8)
project(opencv_candidate)

include(cmake/OpenCVUtils.cmake)

OCV_OPTION(ENABLE_SSE                 "Enable SSE instructions"                                  ON  )
OCV_OPTION(ENABLE_SSE2                "Enable SSE2 instructions"                                 ON  )
OCV_OPTION(ENABLE_SSE3                "Enable SSE3 instructions"                                 OFF )
OCV_OPTION(ENABLE_SSSE3               "Enable SSSE3 instructions"                                ON )
OCV_OPTION(ENABLE_SSE41               "Enable SSE4.1 instructions"                               OFF )
OCV_OPTION(ENABLE_SSE42               "Enable SSE4.2 instructions"                               OFF )

include(cmake/OpenCVCompilerOptions.cmake)

# In case of Makefiles if the user does not setup CMAKE_BUILD_TYPE, assume it's Release:
if(CMAKE_GENERATOR MATCHES "Makefiles|Ninja" AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(catkin REQUIRED)
find_package(libg2o QUIET)
find_package(OpenCV)
find_library(DEPTH_SENSE_LIBRARIES DepthSense /opt/DepthSenseSDK/lib)


if (${DEPTH_SENSE_LIBRARIES} STREQUAL DEPTH_SENSE_LIBRARIES-NOTFOUND)
  set(OPENCV_CREATIVE "")
else()
  set(OPENCV_CREATIVE opencv_creative)
endif()

if (${catkin_VERSION} VERSION_GREATER "0.5.28")
if (libg2o_FOUND)
  set(EXTRA_DEPENDS libg2o)
  set(EXTRA_LIBRARIES opencv_candidate_reconst3d)
endif()
catkin_package(CFG_EXTRAS depends.cmake
               DEPENDS OpenCV ${EXTRA_DEPENDS}
               INCLUDE_DIRS include
               LIBRARIES opencv_candidate
                         ${OPENCV_CREATIVE}
                         opencv_rgbd
			 opencv_line_rgb
                         ${opencv_candidate_reconst3d}
)
else()
catkin_stack()
catkin_project(opencv_candidate
               CFG_EXTRAS depends.cmake
               DEPENDS OpenCV libg2o
               INCLUDE_DIRS include
               LIBRARIES opencv_candidate
                         ${OPENCV_CREATIVE}
                         opencv_rgbd
			 opencv_line_rgb
)
list(APPEND catkin_LIBRARIES ${libg2o_LIBRARIES})
endif()

add_definitions(-Wall)

find_package(OpenCV REQUIRED core imgproc highgui calib3d)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV headers: ${OpenCV_INCLUDE_DIRS}")
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
include_directories(include)

# install the include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/rgbd/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/line_rgb/include)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/opencv_candidate
        DESTINATION include
        COMPONENT main
)

add_subdirectory(src)





# In case of Makefiles if the user does not setup CMAKE_BUILD_TYPE, assume it's Release:
if(CMAKE_GENERATOR MATCHES "Makefiles|Ninja" AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()
