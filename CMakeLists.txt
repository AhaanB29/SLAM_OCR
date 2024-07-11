cmake_minimum_required(VERSION 3.10)

# Project name
project(OCR)

# Set C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(OCR func.cpp)

# Link OpenCV libraries
target_link_libraries(OCR ${OpenCV_LIBS} pthread)
