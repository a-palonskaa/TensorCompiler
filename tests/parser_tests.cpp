#include <gtest/gtest.h>

#include <filesystem>

#include "parser.hpp"
#include "types.hpp"

using namespace TensorCompiler;

TEST(SimpleParserTest, AddMulModel) {
    ONNXParser parser;

    std::string model_path = std::string(PROJECT_ROOT) + "/models/add_mul.onnx";

    std::cout << "Model path: " << model_path << '\n';
    EXPECT_TRUE(std::filesystem::exists(model_path)) << "File does not exist\n";
    ASSERT_TRUE(parser.load(model_path)) << "Load failed\n";

    ASSERT_TRUE(parser.load(model_path))
        << "Couldnot load add_mul.onnx model.\n";

    const auto& graph = parser.ParseGraph();

    EXPECT_EQ(graph.tensors().size(), 5);
    EXPECT_EQ(graph.nodes().size(), 2);

    std::vector<std::string> expected_tensors = {"input1", "input2", "input3",
                                                 "add_out", "output"};
    for (const auto& name : expected_tensors) {
        const Tensor* t = graph.get_const_tensor(name);
        EXPECT_NE(t, nullptr) << "Tensor '" << name << "' not found\n";
    }

    const Node* add_node = nullptr;
    const Node* mul_node = nullptr;
    for (const auto& [name, node] : graph.nodes()) {
        if (node->op_type() == OpType::Add) add_node = node.get();
        if (node->op_type() == OpType::Mul) mul_node = node.get();
    }

    ASSERT_NE(add_node, nullptr) << "Add node not found\n";
    ASSERT_NE(mul_node, nullptr) << "Mul node not found\n";

    EXPECT_EQ(add_node->inputs().size(), 2);
    EXPECT_EQ(add_node->outputs().size(), 1);
    EXPECT_EQ(add_node->inputs()[0], "input1");
    EXPECT_EQ(add_node->inputs()[1], "input2");
    EXPECT_EQ(add_node->outputs()[0], "add_out");

    EXPECT_EQ(mul_node->inputs().size(), 2);
    EXPECT_EQ(mul_node->outputs().size(), 1);
    EXPECT_EQ(mul_node->inputs()[0], "add_out");
    EXPECT_EQ(mul_node->inputs()[1], "input3");
    EXPECT_EQ(mul_node->outputs()[0], "output");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}