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
    const auto& name() const { return name_; }
    const auto& shape() const { return shape_; }
    DataType data_type() const { return data_type_; }
    bool is_constant() const { return is_constant_; }
    const auto& raw_data() const { return raw_data_; }
    const auto& string_data() const { return string_data_; }
    const auto& producer() const { return producer_; }
    const auto& consumers() const { return consumers_; }

    void set_name(const std::string& n) { name_ = n; }
    void set_shape(const std::vector<int64_t>& s) { shape_ = s; }
    void set_data_type(DataType t) { data_type_ = t; }
    void set_is_constant(bool c) { is_constant_ = c; }
    void set_raw_data(const std::vector<uint8_t>& d) { raw_data_ = d; }
    void set_string_data(const std::vector<std::string>& d) {
        string_data_ = d;
    }
    void set_producer(const std::string& p) { producer_ = p; }
    void add_consumer(const std::string& c) { consumers_.push_back(c); }

   private:
    std::string name_;
    std::vector<int64_t> shape_;
    DataType data_type_ = DataType::UNDEFINED;

    bool is_constant_ = false;
    std::vector<uint8_t> raw_data_;
    std::vector<std::string> string_data_;

    std::string producer_;
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

    void ToGraphViz(const std::string& filename) const;
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
