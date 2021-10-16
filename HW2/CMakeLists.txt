cmake_minimum_required(VERSION 3.16)
project(HW2)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/g2o/cmake_modules)
set (G2O_ROOT  /home/user/local/)
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

SET(CMAKE_CXX_STANDARD 14)

add_executable(HW2 main.cpp)
