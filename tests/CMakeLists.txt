include(FetchContent)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)

FetchContent_MakeAvailable(googletest)

add_executable(parser_test
    parser_tests.cpp)

target_link_libraries(parser_test
    PRIVATE
        onnx_parser
        GTest::gtest
)

target_include_directories(parser_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../frontend/parser/onnx/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../frontend/ir/include
        ${CMAKE_BINARY_DIR}
)

include(GoogleTest)
gtest_discover_tests(parser_test)