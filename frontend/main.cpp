#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "logger.hpp"
#include "parser.hpp"

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

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx>" << '\n';
        return 1;
    }

    std::filesystem::create_directories("./logs");
    std::filesystem::create_directories("./images");

    if (!Logger::getInstance().setlogFile("./logs/frontend.log")) {
        std::cout << "Failed to open log file, logs will be printd out to "
                     "std::cerr only \n";
    }

    TensorCompiler::ONNXParser parser;

    std::string path = std::string("./models/") + argv[1];
    if (!parser.load(path)) {
        std::cerr << "Failed to parse ONNX model\n";
        return 1;
    }

    const auto& graph = parser.ParseGraph();
    std::string dot_filename = "./images/" + replace_extension(argv[1], ".dot");
    std::string png_filename = "./images/" + replace_extension(argv[1], ".png");

    std::cout << "\nParsed graph with " << graph.nodes().size() << " nodes and "
              << graph.tensors().size() << " tensors.\n\n";
    parser.dump();

    graph.ToGraphViz(dot_filename);

    std::string cmd = "dot -Tpng " + dot_filename + " -o " + png_filename;
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "Error: failed to render PNG (dot returned " << ret
                  << ").\n";
    } else {
        std::cout << "\nGraph rendered to " << png_filename << '\n';
    }

    return 0;
}
