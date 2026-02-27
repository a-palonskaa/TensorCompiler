#include "../include/graphviz.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "types.hpp"

namespace TensorCompiler {

namespace {

std::string AttributeValueToString(const AttributeValue& value) {
    struct Visitor {
        std::string operator()(std::monostate) const { return "null"; }
        std::string operator()(int64_t v) const { return std::to_string(v); }
        std::string operator()(float v) const { return std::to_string(v); }
        std::string operator()(const std::string& v) const { return v; }
        std::string operator()(const std::vector<int64_t>& v) const {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < v.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << v[i];
            }
            oss << "]";
            return oss.str();
        }
        std::string operator()(const std::vector<float>& v) const {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < v.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << v[i];
            }
            oss << "]";
            return oss.str();
        }
    };
    return std::visit(Visitor{}, value);
}

std::string ShapeToString(const std::vector<int64_t>& shape) {
    if (shape.empty()) return "scalar";
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        if (shape[i] == -1)
            oss << "?";
        else
            oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

std::string OpTypeToString(OpType op) {
    switch (op) {
        case OpType::Add:
            return "Add";
        case OpType::Mul:
            return "Mul";
        case OpType::Conv:
            return "Conv";
        case OpType::Relu:
            return "Relu";
        case OpType::MatMul:
            return "MatMul";
        case OpType::Gemm:
            return "Gemm";
        default:
            return "Unknown";
    }
}

}  // namespace

void Graph::ToGraphViz(const std::string& filename) const {
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "Error: cannot open file " << filename << " for writing\n";
        return;
    }

    ofs << "digraph G {\n"
        << "  rankdir=TB;\n"
        << "  newrank=true;\n"
        << "  nodesep=0.5; ranksep=1.0;\n"
        << "  node [fontname=\"Helvetica\"];\n\n";

    for (const auto& [name, node] : nodes_) {
        std::string color =
            (node->op_type_ == OpType::Unknown) ? "crimson" : "lightblue";
        std::string label = name + "\\n" + OpTypeToString(node->op_type_);

        if (!node->attributes_.empty()) {
            label += "\\n[";
            bool first = true;
            for (const auto& [attr_name, attr] : node->attributes_) {
                if (!first) label += ", ";
                first = false;
                label += attr_name + "=" + AttributeValueToString(attr.value_);
            }
            label += "]";
        }

        ofs << "  \"" << name << "\" [label=\"" << label
            << "\", shape=box, style=filled, fillcolor=" << color << "];\n";
    }

    ofs << "\n";

    for (const auto& [name, tensor] : tensors_) {
        std::string color = tensor->is_constant_ ? "lightgreen" : "yellow";
        std::string shape_str = ShapeToString(tensor->shape_);
        ofs << "  \"T_" << name << "\" [label=\"" << name << "\\n"
            << shape_str
            << "\", shape=ellipse, style=filled, fillcolor=" << color << "];\n";
    }

    ofs << "\n";

    for (const auto& [name, node] : nodes_) {
        for (const auto& out : node->outputs_) {
            auto it = tensors_.find(out);
            if (it == tensors_.end()) {
                std::cerr << "Warning: output tensor '" << out
                          << "' not found in graph\n";
                continue;
            }
            ofs << "  \"" << name << "\" -> \"T_" << out
                << "\" [label=\"produces\", color=red];\n";
        }
    }

    for (const auto& [name, node] : nodes_) {
        for (const auto& in : node->inputs_) {
            if (in.empty()) continue;
            auto it = tensors_.find(in);
            if (it == tensors_.end()) {
                std::cerr << "Warning: input tensor '" << in
                          << "' not found in graph\n";
                continue;
            }
            ofs << "  \"T_" << in << "\" -> \"" << name << "\" [label=\"" << in
                << "\", color=blue];\n";
        }
    }

    ofs << "  legend_text [\n"
        << "    label=<\n"
        << "      <table border=\"0\" cellborder=\"1\" cellspacing=\"0\" cellpadding=\"4\">\n"
        << "        <tr><td colspan=\"2\"><b>Legend</b></td></tr>\n"
        << "        <tr><td>Operator</td><td bgcolor=\"lightblue\"> </td></tr>\n"
        << "        <tr><td>Tensor</td><td bgcolor=\"yellow\"> </td></tr>\n"
        << "        <tr><td>Constant tensor</td><td bgcolor=\"lightgreen\"> </td></tr>\n"
        << "        <tr><td>Unknown</td><td bgcolor=\"crimson\"> </td></tr>\n"
        << "      </table>\n"
        << "    >,\n"
        << "    shape=plaintext\n"
        << "  ];\n";

    ofs << "}\n";
}
} // namespace TensorCompiler
