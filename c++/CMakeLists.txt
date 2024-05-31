cmake_minimum_required(VERSION 3.22)
include(FetchContent)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

project(rlb LANGUAGES CXX C)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA libcamera)
set(ncnn_DIR "/usr/local/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)

IF(LIBCAMERA_FOUND) 
  include_directories(${LIBCAMERA_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS})
  add_definitions(-DFOUND_LIBCAMERA)
else()
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

set(SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
file(GLOB_RECURSE SOURCE_FILES "${SOURCE_DIR}/*.cpp")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

add_executable(main ${SOURCE_FILES})

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG master
)
FetchContent_GetProperties(yaml-cpp)

if(NOT yaml-cpp_POPULATED)
  message(STATUS "Fetching yaml-cpp...")
  FetchContent_Populate(yaml-cpp)
  add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})
endif()

IF(LIBCAMERA_FOUND) 
  target_link_libraries(main PUBLIC ${OpenCV_LIBS} onnxruntime yaml-cpp::yaml-cpp -llccv ncnn)
else()
  target_link_libraries(main PUBLIC ${OpenCV_LIBS} onnxruntime yaml-cpp::yaml-cpp ncnn)
endif()
