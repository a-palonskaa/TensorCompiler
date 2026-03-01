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

namespace Config {

constexpr char LOGS_DIR[] = "./logs/";
constexpr char IMG_DIR[] = "./images/";
constexpr char MODELS[] = "./models/";

constexpr char DOT_EXT[] = ".dot";
constexpr char PNG_EXT[] = ".png";
constexpr char LOG_EXT[] = ".log";

constexpr char LOGFILE[] = "frontend.log";

}  // namespace Config

int main(int argc, const char* argv[]) {
    if ((argc < 2) || (argc > 3) ||
        (argc == 3 && strcmp(argv[2], "-d") != 0)) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx>" << "[optional -d]"
                  << '\n';
        return 0;
    }

    if (!Logger::getInstance().setLogFile(std::string(Config::LOGS_DIR) +
                                          std::string(Config::LOGFILE))) {
        LOG(INFO, "logs will be printed to std::cerr\n");
    }

    std::filesystem::create_directories(Config::LOGS_DIR);
    std::filesystem::create_directories(Config::IMG_DIR);

    std::string dot_filename =
        Config::IMG_DIR + replace_extension(argv[1], Config::DOT_EXT);
    std::string png_filename =
        Config::IMG_DIR + replace_extension(argv[1], Config::PNG_EXT);

    TensorCompiler::ONNXParser parser;
    std::string path = std::string(Config::MODELS) + argv[1];
    if (!parser.load(path)) {
        LOG(ERROR, "Failed to parse ONNX model\n");
        return 1;
    }

    const auto& graph = parser.ParseGraph();
    std::cout << "\nParsed graph with " << graph.nodes().size() << " nodes and "
              << graph.tensors().size() << " tensors.\n";

    if (argc == 3 && strcmp(argv[2], "-d") == 0) {
        parser.dump();
    }

    graph.ToGraphViz(dot_filename);

    std::string cmd = "dot -Tpng " + dot_filename + " -o " + png_filename;
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        LOG(WARNING, "Failed to render PNG (dot returned " +
                         std::to_string(ret) + ").\n");
    } else {
        std::cout << "\nGraph rendered to " << png_filename << '\n';
    }

    return 0;
}
