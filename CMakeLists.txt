cmake_minimum_required(VERSION 3.5)

project(yolov8_infer_segment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_ARCHITECTURES 60 61 75 86 CACHE STRING "CUDA architectures" FORCE)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)


find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

find_package(OpenCV REQUIRED)

# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)
include_directories(/usr/local/TensorRT-8.2.5.1/include/)
link_directories(/usr/local/TensorRT-8.2.5.1/lib/)
set(TENSORRT_ROOT /usr/local/TensorRT-8.2.5.1)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/)
find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
    HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER})
set(LIB ${LIB} ${TENSORRT_LIBRARY})
set(INC ${INC} ${TENSORRT_INCLUDE_DIR})


include_directories(${PROJECT_SOURCE_DIR}/include)

INCLUDE_DIRECTORIES(
     ${CMAKE_CURRENT_SOURCE_DIR}/include
 )

SET(PROC_ALL_FILES ${SRCS})
file(GLOB SOURCE_FILES src/*.cpp)

add_executable(yolov8_infer_segment ${PROJECT_SOURCE_DIR}/main.cpp yolov8.cpp)

target_link_libraries(yolov8_infer_segment ${OpenCV_LIBS})
target_link_libraries(yolov8_infer_segment nvinfer ${CUDA_LIBRARIES})
target_link_libraries(yolov8_infer_segment "${TensorRT-8.2.5.1}")
target_link_libraries(yolov8_infer_segment nvonnxparser)
target_link_libraries(yolov8_infer_segment nvinfer)
target_link_libraries(yolov8_infer_segment nvinfer_plugin)
target_link_libraries(yolov8_infer_segment cudart)
add_definitions(-O2 -pthread)

install(TARGETS yolov8_infer_segment
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
