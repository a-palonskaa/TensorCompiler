#pragma once

#include "onnx.pb.h"
#include <string>
#include <vector>
#include <memory>
#include <optional>


class ONNXParser {
public:
    ONNXParser();
    ~ONNXParser();

    bool load(const std::string& filename);
    void dump() const;

    struct TensorInfo {
        std::string name;
        std::vector<int64_t> dimensions;
        int dataType;
    };

    struct NodeInfo {
        std::string opType;
        std::string name;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
    };

    std::string getModelVersion() const;
    std::string getProducerName() const;
    std::vector<TensorInfo> getInputs() const;
    std::vector<TensorInfo> getOutputs() const;
    std::vector<NodeInfo> getNodes() const;
    std::optional<std::vector<float>> getWeights(const std::string& tensorName) const;
private:
    onnx::ModelProto* model_;
    // Graph* graph_;
};
