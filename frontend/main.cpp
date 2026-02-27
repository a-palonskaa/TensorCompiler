#include <cstdlib>
#include <iostream>
#include <string>

#include "parser/onnx/include/parser.hpp"

namespace {

std::string replace_extension(const std::string& filename,
                              const std::string& new_ext) {
    size_t pos = filename.find_last_of('.');
    if (pos == std::string::npos) {
        return filename + new_ext;
    }
    return filename.substr(0, pos) + new_ext;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx>" << '\n';
        return 1;
    }

    TensorCompiler::ONNXParser parser;

    std::string path = std::string("./models/") + argv[1];
    if (!parser.load(path)) {
        std::cerr << "Failed to parse ONNX model\n";
        return 1;
    }

#ifndef NDEBUG
    std::cerr << "Calling ParseGraph... \n";
#endif  // NDEBUG
    const auto& graph = parser.ParseGraph();
    std::string dot_filename = "./images/" + replace_extension(argv[1], ".dot");
    std::string png_filename = "./images/" + replace_extension(argv[1], ".png");

#ifndef NDEBUG
    std::cerr << "Calling ToGraphViz...\n";
#endif  // NDEBUG
    graph.ToGraphViz(dot_filename);

    std::cout << "\nParsed graph with " << graph.nodes_.size() << " nodes and "
              << graph.tensors_.size() << " tensors.\n\n";
    parser.dump();

    std::string cmd = "dot -Tpng " + dot_filename + " -o " + png_filename;
    std::system(cmd.c_str());

    std::cout << "\nGraph rendered to " << png_filename << '\n';

    return 0;
}
