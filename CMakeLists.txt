cmake_minimum_required(VERSION 3.21)

project(TensorCompiler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf REQUIRED)

set(ONNX_PROTO "${CMAKE_CURRENT_SOURCE_DIR}/onnx.proto")

protobuf_generate_cpp(
    ONNX_PROTO_SRCS
    ONNX_PROTO_HDRS
    ${ONNX_PROTO}
)

add_library(onnx_parser
    STATIC
        frontend/parser/onnx/src/parser.cpp
        ${ONNX_PROTO_SRCS}
)

target_include_directories(onnx_parser
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/frontend/parser/onnx/include
        ${CMAKE_CURRENT_SOURCE_DIR}/frontend/ir/include
        ${Protobuf_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(onnx_parser
    PRIVATE
        ${Protobuf_LIBRARIES}
)

add_executable(run
    frontend/main.cpp)

target_include_directories(run
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/frontend/parser/onnx/include
        ${CMAKE_CURRENT_SOURCE_DIR}/frontend/ir/include
        ${Protobuf_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(run
    PRIVATE
        onnx_parser
)

add_subdirectory(flags)

#enable_testing()

#add_subdirectory(tests)