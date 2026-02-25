#include "parser.hpp"
#include "onnx.pb.h"
#include <fstream>
#include <iostream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

ONNXParser::ONNXParser() {
    model_ = new onnx::ModelProto();
    GOOGLE_PROTOBUF_VERIFY_VERSION;
}

ONNXParser::~ONNXParser() {
    delete model_;
    google::protobuf::ShutdownProtobufLibrary();
}

bool ONNXParser::load(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        std::cerr << "failed to open file: " << filename << "\n"; //TODO - logger
        return false;
    }

    if (!model_->ParseFromIstream(&input)) {
        std::cerr << "failed to parse model_" << "\n"; //TODO - logger
        return false;
    }
    return true;
}

std::string ONNXParser::getModelVersion() const {
    if (model_ && model_->has_model_version()) {
        return std::to_string(model_->model_version());
    }
    return "unknown";
}

std::string ONNXParser::getProducerName() const {
    if (model_ && model_->has_producer_name()) {
        return model_->producer_name();
    }
    return "unknown";
}

std::vector<ONNXParser::TensorInfo> ONNXParser::getInputs() const {
    std::vector<ONNXParser::TensorInfo> inputs;
    if (!model_ || !model_->has_graph()) return inputs;
    const auto& graph = model_->graph();

    for (const auto& input : graph.input()) {
        ONNXParser::TensorInfo info;
        info.name = input.name();
        if (input.has_type() && input.type().has_tensor_type()) {
            const auto& tensor = input.type().tensor_type();
            info.dataType = tensor.elem_type();
            if (tensor.has_shape()) {
                for (const auto& dim : tensor.shape().dim()) {
                    if (dim.has_dim_value()) {
                        info.dimensions.push_back(dim.dim_value());
                    } else {
                        info.dimensions.push_back(-1);
                    }
                }
            }
        }
        inputs.push_back(info);
    }
    return inputs;
}

std::vector<ONNXParser::TensorInfo> ONNXParser::getOutputs() const {
    std::vector<ONNXParser::TensorInfo> outputs;
    if (!model_ || !model_->has_graph()) return outputs;
    const auto& graph = model_->graph();
    for (const auto& output : graph.output()) {
        ONNXParser::TensorInfo info;
        info.name = output.name();
        if (output.has_type() && output.type().has_tensor_type()) {
            const auto& tensor = output.type().tensor_type();
            info.dataType = tensor.elem_type();
            if (tensor.has_shape()) {
                for (const auto& dim : tensor.shape().dim()) {
                    if (dim.has_dim_value()) {
                        info.dimensions.push_back(dim.dim_value());
                    } else {
                        info.dimensions.push_back(-1);
                    }
                }
            }
        }
        outputs.push_back(info);
    }
    return outputs;
}

std::vector<ONNXParser::NodeInfo> ONNXParser::getNodes() const {
    std::vector<ONNXParser::NodeInfo> nodes;
    if (!model_ || !model_->has_graph()) return nodes;
    const auto& graph = model_->graph();
    for (const auto& node : graph.node()) {
        ONNXParser::NodeInfo info;
        info.opType = node.op_type();
        info.name = node.name();
        for (const auto& input : node.input()) {
            info.inputs.push_back(input);
        }
        for (const auto& output : node.output()) {
            info.outputs.push_back(output);
        }
        nodes.push_back(info);
    }
    return nodes;
}

std::optional<std::vector<float>> ONNXParser::getWeights(const std::string& tensorName) const {
    if (!model_ || !model_->has_graph()) return std::nullopt;
    const auto& graph = model_->graph();
    for (const auto& init : graph.initializer()) {
        if (init.name() == tensorName) {
            std::vector<float> weights;
            if (init.data_type() == 1) { // 1 = FLOAT // ХУЙНЯ -
                const char* data = init.raw_data().data();
                size_t size = init.raw_data().size() / sizeof(float);
                weights.resize(size);
                std::memcpy(weights.data(), data, init.raw_data().size());
            }
            return weights;
        }
    }
    return std::nullopt;
}

void ONNXParser::dump() const {
    if (!model_) {
        std::cout << "No model loaded\n";
        return;
    }

    std::cout << "=== ONNX Model Info ===\n";
    std::cout << "Producer: " << getProducerName() << "\n";
    std::cout << "Version: " << getModelVersion() << "\n";

    if (model_->has_graph()) {
        const auto& graph = model_->graph();
        std::cout << "Graph name: " << graph.name() << "\n";
        auto inputs = getInputs();
        std::cout << "\nInputs (" << inputs.size() << "):\n";
        for (const auto& input : inputs) {
            std::cout << "  - " << input.name << " [";
            for (size_t i = 0; i < input.dimensions.size(); ++i) {
                if (i > 0) std::cout << ", ";
                if (input.dimensions[i] == -1) {
                    std::cout << "?";
                } else {
                    std::cout << input.dimensions[i];
                }
            }
            std::cout << "]\n";
        }
        auto nodes = getNodes();
        std::cout << "\nOperators (" << nodes.size() << "):\n";
        for (const auto& node : nodes) {
            std::cout << "  - " << node.opType;
            if (!node.name.empty()) {
                std::cout << " (" << node.name << ")";
            }
            std::cout << "\n";
        }
    }
}
