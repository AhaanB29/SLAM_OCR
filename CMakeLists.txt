cmake_minimum_required(VERSION 3.10)
project(SLAM_OCR)

set(CMAKE_CXX_STANDARD 11)
# Find packages
find_package(OpenCV REQUIRED)
# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
# Link libraries
add_executable(ocr func.cpp)
target_link_libraries(ocr ${OpenCV_LIBS})
target_link_libraries(ocr -ltesseract -llept)
