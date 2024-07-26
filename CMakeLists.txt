cmake_minimum_required(VERSION 3.2)
project(SLAM_OCR)

find_package(OpenCV 4.6 REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executables and link libraries
add_executable(ocr func.cpp)
target_link_libraries(ocr ${OpenCV_LIBS} tesseract lept)

add_executable(digits digit_recog.cpp)
target_link_libraries(digits ${OpenCV_LIBS})
 
set(PROJECT_SOURCES
    main.cpp
    inference.h
    inference.cpp
)
add_executable(yolo ${PROJECT_SOURCES})
target_link_libraries(yolo ${OpenCV_LIBS})
