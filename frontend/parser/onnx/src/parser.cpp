#include "parser.hpp"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <iostream>

#include "converters.hpp"
#include "logger.hpp"
#include "onnx.pb.h"

namespace TensorCompiler {

ONNXParser::ONNXParser()
    : model_(std::make_unique<onnx::ModelProto>()),
      graph_(std::make_unique<Graph>()) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
}

ONNXParser::~ONNXParser() { google::protobuf::ShutdownProtobufLibrary(); }

bool ONNXParser::load(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        LOG(ERROR, "failed to open file: " + filename);
        return false;
    }

    if (!model_->ParseFromIstream(&input)) {
        LOG(ERROR, "failed to parse model");
        return false;
    }
    return true;
}

std::string ONNXParser::getModelVersion() const {
    if (model_ && model_->has_model_version()) {
        return std::to_string(model_->model_version());
    }
    LOG(INFO, "unknown version of a model");
    return "unknown";
}

std::string ONNXParser::getProducerName() const {
    if (model_ && model_->has_producer_name()) {
        return model_->producer_name();
    }
    LOG(INFO, "unknown producer name");
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

std::optional<TensorData> ONNXParser::getWeights(
    const std::string& tensorName) const {
    if (!model_ || !model_->has_graph()) return std::nullopt;

    const auto& graph = model_->graph();
    for (const auto& init : graph.initializer()) {
        if (init.name() != tensorName) continue;

        switch (init.data_type()) {
            case onnx::TensorProto_DataType_FLOAT: {
                std::vector<float> data;
                if (init.has_raw_data()) {
                    size_t size = init.raw_data().size() / sizeof(float);
                    data.resize(size);
                    std::memcpy(data.data(), init.raw_data().data(),
                                init.raw_data().size());
                } else {
                    data.assign(init.float_data().begin(),
                                init.float_data().end());
                }
                return data;
            }
            case onnx::TensorProto_DataType_DOUBLE: {
                std::vector<double> data;
                if (init.has_raw_data()) {
                    size_t size = init.raw_data().size() / sizeof(double);
                    data.resize(size);
                    std::memcpy(data.data(), init.raw_data().data(),
                                init.raw_data().size());
                } else {
                    data.assign(init.double_data().begin(),
                                init.double_data().end());
                }
                return data;
            }
            case onnx::TensorProto_DataType_INT32: {
                std::vector<int32_t> data;
                if (init.has_raw_data()) {
                    size_t size = init.raw_data().size() / sizeof(int32_t);
                    data.resize(size);
                    std::memcpy(data.data(), init.raw_data().data(),
                                init.raw_data().size());
                } else {
                    data.assign(init.int32_data().begin(),
                                init.int32_data().end());
                }
                return data;
            }
            case onnx::TensorProto_DataType_INT64: {
                std::vector<int64_t> data;
                if (init.has_raw_data()) {
                    size_t size = init.raw_data().size() / sizeof(int64_t);
                    data.resize(size);
                    std::memcpy(data.data(), init.raw_data().data(),
                                init.raw_data().size());
                } else {
                    data.assign(init.int64_data().begin(),
                                init.int64_data().end());
                }
                return data;
            }
            default:
                return std::nullopt;
        }
    }
    return std::nullopt;
}

void ONNXParser::dump() const {
    if (!model_) {
        LOG(ERROR, "cannot dump, no model loaded");
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

const Graph& ONNXParser::ParseGraph() {
    if (graph_parsed_) return *graph_;

    graph_ = std::make_unique<Graph>();
    Graph& graph = *graph_;

    if (!model_ || !model_->has_graph()) return graph;
    const auto& onnx_graph = model_->graph();

    for (const auto& input : onnx_graph.input()) {
        auto tensor = ConvertTensorFromValueInfo(input);
        graph.add_tensor(std::move(tensor));
        graph.add_input(input.name());
    }

    for (const auto& output : onnx_graph.output()) {
        auto tensor = ConvertTensorFromValueInfo(output);
        graph.add_tensor(std::move(tensor));
        graph.add_output(output.name());
    }

    for (const auto& init : onnx_graph.initializer()) {
        graph.add_tensor(ConvertTensorFromInitializer(init));
    }

    for (const auto& value_info : onnx_graph.value_info()) {
        if (!graph.get_tensor(value_info.name())) {
            graph.add_tensor(ConvertTensorFromValueInfo(value_info));
        }
    }

    for (const auto& node_proto : onnx_graph.node()) {
        for (const auto& out : node_proto.output()) {
            if (!graph.get_tensor(out)) {
                auto tensor = std::make_unique<Tensor>();
                tensor->set_name(out);
                tensor->set_data_type(DataType::UNDEFINED);
                graph.add_tensor(std::move(tensor));
            }
        }
    }

    for (const auto& node_proto : onnx_graph.node()) {
        auto node = ConvertNode(node_proto);
        std::string node_name = node->name();
        Node* node_ptr = graph.add_node(std::move(node));

        for (const auto& out : node_ptr->outputs()) {
            if (auto* tensor = graph.get_tensor(out)) {
                tensor->set_producer(node_name);
            } else {
                LOG(ERROR, "Output tensor " + out + " not found for node " +
                               node_name);
            }
        }

        for (const auto& in : node_ptr->inputs()) {
            if (in.empty()) continue;
            if (auto* tensor = graph.get_tensor(in)) {
                tensor->add_consumer(node_name);
            } else {
                LOG(ERROR,
                    "Input tensor " + in + " not found for node " + node_name);
            }
        }
    }

    graph_parsed_ = true;
    return graph;
}

}  // namespace TensorCompiler
