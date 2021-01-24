cmake_minimum_required(VERSION 3.0)
set(LNAME DenseTracking)
project(${LNAME} LANGUAGES CXX CUDA)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_library(${LNAME}
SHARED
src/ImageProc.cu
src/DenseTracking.cpp
src/DeviceFunction.cu)

set_target_properties(${LNAME}
PROPERTIES 
CUDA_SEPARABLE_COMPILATION ON
POSITION_INDEPENDENT_CODE ON
CUDA_STANDARD 11
CUDA_ARCHITECTURES 86)

target_include_directories(${LNAME}
PUBLIC
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
$<INSTALL_INTERFACE:include/DenseTracking>)


target_link_libraries(${LNAME}
PUBLIC
Eigen3::Eigen
${OpenCV_LIBS})

add_subdirectory(test)

install(
TARGETS ${LNAME}
EXPORT ${LNAME}-export
LIBRARY DESTINATION lib
ARCHIVE DESTINATION lib
CONFIGURATIONS Release)

install(
EXPORT ${LNAME}-export
FILE ${LNAME}Config.cmake
NAMESPACE XYUTILS::
DESTINATION lib/cmake/${LNAME})

install(
DIRECTORY "${CMAKE_SOURCE_DIR}/include/"
DESTINATION "include/${LNAME}"
FILES_MATCHING
PATTERN "*.h")