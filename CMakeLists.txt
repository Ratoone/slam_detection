cmake_minimum_required(VERSION 3.15)
project(slam_detection)

set(CMAKE_CXX_STANDARD 17)

include_directories(./include)
find_package( OpenCV REQUIRED )

add_executable(slam_detection main.cpp src/Calibration.cpp src/VideoStream.cpp src/Matching.cpp src/ObjectDetection.cpp)
target_link_libraries( slam_detection ${OpenCV_LIBS} )