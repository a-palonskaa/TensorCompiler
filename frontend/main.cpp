#include "parser/onnx/include/parser.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx>" << std::endl;
        return 1;
    }

    ONNXParser parser;

    if (!parser.load(argv[1])) {
        std::cerr << "Failed to parse ONNX model\n";
        return 1;
    }

    parser.dump();

    return 0;
}
