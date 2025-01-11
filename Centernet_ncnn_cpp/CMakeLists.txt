cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_STANDARD 17)
set(ncnn_DIR /mnt/e/ncnn/build)
set(ncnn_LIB /mnt/e/ncnn/build/src)
project(nanodet_demo)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV REQUIRED)

#find_package(ncnn REQUIRED)
# if(NOT TARGET ncnn)
#     message(WARNING "ncnn NOT FOUND!  Please set ncnn_DIR environment variable")
# else()
#     message("ncnn FOUND ")
# endif()

include_directories(
    /mnt/e/ncnn/src
    /mnt/e/ncnn/build/src
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)


add_executable(nanodet_demo main.cpp)

target_link_libraries(
    nanodet_demo
    ${OpenCV_LIBS}
    "/mnt/e/ncnn/build/src/libncnn.a"
)
