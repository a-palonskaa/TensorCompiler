#ifndef CONVERTERS_HPP_
#define CONVERTERS_HPP_

#include <fstream>
#include <iostream>

#include "../include/parser.hpp"
#include "onnx.pb.h"
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
    return OpType::Unknown;
}

std::unique_ptr<Tensor> ConvertTensorFromValueInfo(
    const onnx::ValueInfoProto& value_info) {
    auto tensor = std::make_unique<Tensor>();
    tensor->name_ = value_info.name();

    if (value_info.has_type() && value_info.type().has_tensor_type()) {
        const auto& tensor_type = value_info.type().tensor_type();
        tensor->data_type_ = ConvertDataType(tensor_type.elem_type());

        if (tensor_type.has_shape()) {
            for (const auto& dim : tensor_type.shape().dim()) {
                tensor->shape_.push_back(dim.has_dim_value() ? dim.dim_value()
                                                             : -1);
            }
        }
        return tensor;
    }
    // TODO - log добавить
    return tensor;
}

std::unique_ptr<Tensor> ConvertTensorFromInitializer(
    const onnx::TensorProto& init) {
    auto tensor = std::make_unique<Tensor>();
    tensor->name_ = init.name();
    tensor->data_type_ = ConvertDataType(init.data_type());
    tensor->shape_.assign(init.dims().begin(), init.dims().end());
    tensor->is_constant_ = true;
    if (init.has_raw_data()) {
        tensor->raw_data_.assign(init.raw_data().begin(),
                                 init.raw_data().end());
        return tensor;
    }  // else { //TODO - обработка других типов }
    // TODO - log про отсутствующий тип?
    return tensor;
}

Attribute ConvertAttribute(const onnx::AttributeProto& attr_proto) {
    Attribute attr;
    attr.name_ = attr_proto.name();
    switch (attr_proto.type()) {
        case onnx::AttributeProto_AttributeType_INT:
            attr.value_ = attr_proto.i();
            break;
        case onnx::AttributeProto_AttributeType_FLOAT:
            attr.value_ = attr_proto.f();
            break;
        case onnx::AttributeProto_AttributeType_STRING:
            attr.value_ = attr_proto.s();
            break;
        case onnx::AttributeProto_AttributeType_INTS:
            attr.value_ = std::vector<int64_t>(attr_proto.ints().begin(),
                                               attr_proto.ints().end());
            break;
        case onnx::AttributeProto_AttributeType_FLOATS:
            attr.value_ = std::vector<float>(attr_proto.floats().begin(),
                                             attr_proto.floats().end());
            break;
        default:
            attr.value_ = std::monostate{};
            break;
    }
    return attr;
}

std::unique_ptr<Node> ConvertNode(const onnx::NodeProto& node_proto) {
    auto node = std::make_unique<Node>();
    node->op_type_ = ConvertOpType(node_proto.op_type());

    if (node_proto.name().empty()) {
        static int counter = 0;
        node->name_ = "node_" + std::to_string(++counter);
    } else {
        node->name_ = node_proto.name();
    }

    node->domain_ = node_proto.domain();
    node->inputs_.assign(node_proto.input().begin(), node_proto.input().end());
    node->outputs_.assign(node_proto.output().begin(),
                          node_proto.output().end());

    for (const auto& attr_proto : node_proto.attribute()) {
        auto attr = ConvertAttribute(attr_proto);
        node->attributes_[attr.name_] = attr;
    }

    return node;
}

}  // namespace TensorCompiler

#endif  // CONVERTERS_HPP_