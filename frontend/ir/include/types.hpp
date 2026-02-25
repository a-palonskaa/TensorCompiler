// NOTE - супер условно в качестве кея потом наверно стоит брать id сгенеренный
#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace TensorCompiler {

using AttributeValue = std::variant<std::monostate, int64_t, float, std::string,
                                    std::vector<int64_t>, std::vector<float>>;

enum class OpType { Add, Mul, Conv, Relu, MatMul, Gemm, Unknown };

enum class DataType {
    BOOL,
    INT8,
    INT16,
    INT32,
    INT64,
    FLOAT,
    STRING,
    UNDEFINED
};

class Tensor {
   public:
    std::string name_;
    std::vector<int64_t> shape_;
    DataType data_type_;

    bool is_constant_ = false;
    std::vector<uint8_t> raw_data_;

    std::string producer_ = 0;
    std::vector<std::string> consumers_;
};

class Attribute {
   public:
    std::string name_;
    AttributeValue value_;
};

class Node {
   public:
    OpType op_type_;
    std::string name_;
    std::string domain_;

    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;

    std::unordered_map<std::string, Attribute> attributes_;
};

class Graph {
   public:
    std::unordered_map<std::string, std::unique_ptr<Tensor>> tensors_;
    std::unordered_map<std::string, std::unique_ptr<Node>> nodes_;

    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
};

}  // namespace TensorCompiler

#endif  // TYPES_HPP_

/*
1. добавляем все input tensors
2. добавляем все output tensors
3. добавляем все initialisers tensors
4. проходим по узлам и собираем все имена выходных тензоров
5. проходим по node - создаем node, подвязываем все входны(уже существуют), все
выходные(-//-) и в тензоры записываем producer, consumer
6. проходим по nodes-initializers(веса) - нет входов, только выходы, выходныне
тензоры запоминаю как producer
*/