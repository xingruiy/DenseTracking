cmake_minimum_required(VERSION 3.0)
project(test_prog CXX)

add_executable(test_prog test_prog.cpp)
target_link_libraries(test_prog PUBLIC DenseTracking)