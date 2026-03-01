cmake_minimum_required(VERSION 3.21)

project(TensorCompiler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf REQUIRED)
find_package(absl REQUIRED)

add_compile_definitions(PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

set(ONNX_PROTO "${CMAKE_CURRENT_SOURCE_DIR}/onnx.proto")

protobuf_generate_cpp(
    ONNX_PROTO_SRCS
    ONNX_PROTO_HDRS
    ${ONNX_PROTO}
)

add_library(onnx_parser
    STATIC
        frontend/parser/onnx/src/parser.cpp
        frontend/graphviz/src/graphviz.cpp
        ${ONNX_PROTO_SRCS}
)

target_include_directories(onnx_parser
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/frontend/parser/onnx/include
        ${CMAKE_CURRENT_SOURCE_DIR}/frontend/ir/include
        ${CMAKE_CURRENT_SOURCE_DIR}/frontend/graphviz/include
        ${CMAKE_CURRENT_SOURCE_DIR}/common/logger
        ${Protobuf_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(onnx_parser
    PUBLIC
        ${Protobuf_LIBRARIES}
        ${Protobuf_TARGETS}
        absl::log_internal_check_op
        absl::log_internal_message
        absl::log_internal_nullguard
        absl::log_internal_proto
        absl::log_initialize
        absl::log_severity
)

add_executable(run
    frontend/main.cpp
)

target_include_directories(run
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/frontend/parser/onnx/include
        ${CMAKE_CURRENT_SOURCE_DIR}/frontend/ir/include
        ${CMAKE_CURRENT_SOURCE_DIR}/frontend/graphviz/include
        ${CMAKE_CURRENT_SOURCE_DIR}/common/logger
        ${Protobuf_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(run
    PRIVATE
        onnx_parser
        ${Protobuf_TARGETS}
        absl::log_internal_check_op
        absl::log_internal_message
        absl::log_internal_nullguard
        absl::log_internal_proto
        absl::log_initialize
        absl::log_severity
)

add_subdirectory(flags)

enable_testing()
add_subdirectory(tests)
