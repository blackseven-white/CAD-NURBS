cmake_minimum_required(VERSION 3.16)

project(CADNURBS VERSION 1.0)
include_directories(${PROJECT_SOURCE_DIR}/3rdparty)
include_directories(${PROJECT_SOURCE_DIR}/include)

aux_source_directory (src SRC_LIST)
add_executable(CADNURBS ${SRC_LIST})

