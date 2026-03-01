#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "onnx.pb.h"
#include "types.hpp"

using TensorData = std::variant<std::vector<float>, std::vector<double>,
                                std::vector<int32_t>, std::vector<int64_t>>;

namespace TensorCompiler {

class ONNXParser {
   public:
    ONNXParser();
    ~ONNXParser();

    struct TensorInfo {
        std::string name;
        std::vector<int64_t> dimensions;
        int dataType = 0;
    };

    struct NodeInfo {
        std::string opType;
        std::string name;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
    };

    bool load(const std::string& filename);

    std::string getModelVersion() const;
    std::string getProducerName() const;
    std::vector<TensorInfo> getInputs() const;
    std::vector<TensorInfo> getOutputs() const;
    std::vector<NodeInfo> getNodes() const;
    std::optional<TensorData> getWeights(const std::string& tensorName) const;

    void dump() const;
    const Graph& ParseGraph();

   private:
    std::unique_ptr<onnx::ModelProto> model_;
    std::unique_ptr<Graph> graph_;

    bool graph_parsed_ = false;
};

}  // namespace TensorCompiler
