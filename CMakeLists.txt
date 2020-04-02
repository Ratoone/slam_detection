cmake_minimum_required(VERSION 3.15)
project(slam_detection)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)
find_package( OpenCV REQUIRED )

add_executable(slam_detection main.cpp)
target_link_libraries( slam_detection ${OpenCV_LIBS} )