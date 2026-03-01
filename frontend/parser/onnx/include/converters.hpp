#ifndef CONVERTERS_HPP_
#define CONVERTERS_HPP_

#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>

#include "logger.hpp"
#include "onnx.pb.h"
#include "parser.hpp"
#include "types.hpp"

namespace TensorCompiler {

DataType ConvertDataType(int onnx_type) {
    switch (onnx_type) {
        case onnx::TensorProto_DataType_BOOL:
            return DataType::BOOL;
        case onnx::TensorProto_DataType_INT8:
            return DataType::INT8;
        case onnx::TensorProto_DataType_INT16:
            return DataType::INT16;
        case onnx::TensorProto_DataType_INT32:
            return DataType::INT32;
        case onnx::TensorProto_DataType_INT64:
            return DataType::INT64;
        case onnx::TensorProto_DataType_FLOAT:
            return DataType::FLOAT;
        case onnx::TensorProto_DataType_STRING:
            return DataType::STRING;
        default:
            LOG(WARNING,
                "Unsupported ONNX data type: " + std::to_string(onnx_type));
            return DataType::UNDEFINED;
    }
}

OpType ConvertOpType(const std::string& op_type_str) {
    if (op_type_str == "Add") return OpType::Add;
    if (op_type_str == "Mul") return OpType::Mul;
    if (op_type_str == "Conv") return OpType::Conv;
    if (op_type_str == "Relu") return OpType::Relu;
    if (op_type_str == "MatMul") return OpType::MatMul;
    if (op_type_str == "Gemm") return OpType::Gemm;

    LOG(WARNING, "Unknown operation type: " + op_type_str);
    return OpType::Unknown;
}

std::unique_ptr<Tensor> ConvertTensorFromValueInfo(
    const onnx::ValueInfoProto& value_info) {
    auto tensor = std::make_unique<Tensor>();
    tensor->set_name(value_info.name());

    if (value_info.has_type() && value_info.type().has_tensor_type()) {
        const auto& tensor_type = value_info.type().tensor_type();
        tensor->set_data_type(ConvertDataType(tensor_type.elem_type()));

        if (tensor_type.has_shape()) {
            std::vector<int64_t> shape;
            for (const auto& dim : tensor_type.shape().dim()) {
                shape.push_back(dim.has_dim_value() ? dim.dim_value() : -1);
            }
            tensor->set_shape(shape);
        }
    } else {
        LOG(WARNING, "ValueInfo '" + value_info.name() +
                         "' has no tensor type information");
    }
    return tensor;
}

std::unique_ptr<Tensor> ConvertTensorFromInitializer(
    const onnx::TensorProto& init) {
    auto tensor = std::make_unique<Tensor>();
    tensor->set_name(init.name());
    tensor->set_data_type(ConvertDataType(init.data_type()));
    tensor->set_shape(
        std::vector<int64_t>(init.dims().begin(), init.dims().end()));
    tensor->set_is_constant(true);

    size_t elem_size = 0;
    switch (init.data_type()) {
        case onnx::TensorProto_DataType_FLOAT:
        case onnx::TensorProto_DataType_INT32:
            elem_size = 4;
            break;
        case onnx::TensorProto_DataType_INT64:
            elem_size = 8;
            break;
        case onnx::TensorProto_DataType_INT8:
        case onnx::TensorProto_DataType_BOOL:
            elem_size = 1;
            break;
        case onnx::TensorProto_DataType_INT16:
            elem_size = 2;
            break;
        case onnx::TensorProto_DataType_STRING:
            elem_size = 0;
            break;
        default:
            LOG(WARNING, "Unsupported data type in initializer '" +
                             init.name() + "', raw_data may be incorrect.");
            elem_size = 1;
            break;
    }

    if (init.has_raw_data()) {
        const auto& raw = init.raw_data();
        tensor->set_raw_data(std::vector<uint8_t>(raw.begin(), raw.end()));

        if (!init.dims().empty()) {
            int64_t num_elements =
                std::accumulate(init.dims().begin(), init.dims().end(), 1LL,
                                std::multiplies<int64_t>());
            int64_t expected_bytes = num_elements * elem_size;
            if (static_cast<int64_t>(init.raw_data().size()) !=
                expected_bytes) {
                LOG(WARNING, "Raw data size mismatch for '" + init.name() +
                                 "': expected " +
                                 std::to_string(expected_bytes) +
                                 " bytes, got " +
                                 std::to_string(init.raw_data().size()));
            }
        }
    } else if (init.data_type() == onnx::TensorProto_DataType_STRING) {
        std::vector<std::string> string_data;
        string_data.reserve(init.string_data_size());
        for (int i = 0; i < init.string_data_size(); ++i) {
            string_data.push_back(init.string_data(i));
        }
        tensor->set_string_data(std::move(string_data));
    } else {
        std::vector<uint8_t> raw_data;
        if (init.float_data_size() > 0) {
            raw_data.resize(init.float_data_size() * sizeof(float));
            std::memcpy(raw_data.data(), init.float_data().data(),
                        raw_data.size());
            tensor->set_raw_data(std::move(raw_data));
        } else if (init.int64_data_size() > 0) {
            raw_data.resize(init.int64_data_size() * sizeof(int64_t));
            std::memcpy(raw_data.data(), init.int64_data().data(),
                        raw_data.size());
            tensor->set_raw_data(std::move(raw_data));
        } else if (init.int32_data_size() > 0) {
            raw_data.resize(init.int32_data_size() * sizeof(int32_t));
            std::memcpy(raw_data.data(), init.int32_data().data(),
                        raw_data.size());
            tensor->set_raw_data(std::move(raw_data));
        }
    }
    return tensor;
}

Attribute ConvertAttribute(const onnx::AttributeProto& attr_proto) {
    Attribute attr;
    attr.set_name(attr_proto.name());
    switch (attr_proto.type()) {
        case onnx::AttributeProto_AttributeType_INT:
            attr.set_value(attr_proto.i());
            break;
        case onnx::AttributeProto_AttributeType_FLOAT:
            attr.set_value(attr_proto.f());
            break;
        case onnx::AttributeProto_AttributeType_STRING:
            attr.set_value(attr_proto.s());
            break;
        case onnx::AttributeProto_AttributeType_INTS:
            attr.set_value(std::vector<int64_t>(attr_proto.ints().begin(),
                                                attr_proto.ints().end()));
            break;
        case onnx::AttributeProto_AttributeType_FLOATS:
            attr.set_value(std::vector<float>(attr_proto.floats().begin(),
                                              attr_proto.floats().end()));
            break;
        default:
            LOG(WARNING, "Unsupported attribute type " +
                             std::to_string(attr_proto.type()) +
                             " for attribute '" + attr_proto.name() + "'");
            attr.set_value(std::monostate{});
            break;
    }
    return attr;
}

std::unique_ptr<Node> ConvertNode(const onnx::NodeProto& node_proto) {
    auto node = std::make_unique<Node>();
    node->set_op_type(ConvertOpType(node_proto.op_type()));

    if (node_proto.name().empty()) {
        static int counter = 0;
        node->set_name("node_" + std::to_string(++counter));
    } else {
        node->set_name(node_proto.name());
    }

    node->set_domain(node_proto.domain());
    node->set_inputs(std::vector<std::string>(node_proto.input().begin(),
                                              node_proto.input().end()));
    node->set_outputs(std::vector<std::string>(node_proto.output().begin(),
                                               node_proto.output().end()));

    for (const auto& attr_proto : node_proto.attribute()) {
        auto attr = ConvertAttribute(attr_proto);
        node->add_attribute(attr.name(), attr);
    }

    return node;
}

}  // namespace TensorCompiler

#endif  // CONVERTERS_HPP_
