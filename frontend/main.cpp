#include <cstdlib>
#include <iostream>

#include "parser/onnx/include/parser.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx>" << '\n';
        return 1;
    }

    TensorCompiler::ONNXParser parser;

    if (!parser.load(argv[1])) {
        std::cerr << "Failed to parse ONNX model\n";
        return 1;
    }

#ifndef NDEBUG
    std::cerr << "Calling ParseGraph... \n";
#endif  // NDEBUG
    auto graph = parser.ParseGraph();

#ifndef NDEBUG
    std::cerr << "Calling ToGraphViz...\n";
#endif  // NDEBUG
    graph.ToGraphViz("model.dot");

    std::cout << "\nParsed graph with " << graph.nodes_.size() << " nodes and "
              << graph.tensors_.size() << " tensors.\n\n";
    parser.dump();

    system("dot -Tpng model.dot -o model.png");
    std::cout << "\nGraph rendered to model.png" << '\n';

    return 0;
}
