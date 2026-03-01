# ***Tensor Compiler***
>*PROJ_FRONT, Tensor Compiler course 26*

A frontend for a tensor compiler that parses neural network models in ONNX format and converts them into an internal graph representation. The graph can be visualized using Graphviz.

## Frontend features
- Reads **ONNX models** (`.onnx`) using Protocol Buffers;
- Builds a computational graph with tensors and operations;
- Supports operations: *Add*, *Mul*, *Conv*, *Relu*, *MatMul*, *Gemm*;
- Visualizes the graph as a DOT file and renders it to PNG (requires **Graphviz**);
- Logging system with configurable log file (can be found in `./logs`);
- Modular design with clear separation of concerns (parser, converters, graph, visualization).

## Requirements

- C++17
- CMake 3.10+
- Protobuf compiler (`protoc`) and library (`libprotobuf-dev`)
- Graphviz
- Python 3 with `onnx` and `numpy` (optional, for generating test models)

## Installation
```bash
git clone https://github.com/a-palonskaa/TensorCompiler
cd TensorCompiler
```

## Build

```bash
cmake -S . -B build -DCMAKE-COMPILE-OPTIONS=ON
cmake --build build --parallel
```

## Run

```bash
./build/run [name].onnx
```
Name from `./models`.
Image of graph and a dot file are located in `./images`.

### Notes
To get absl:
```bash
git clone https://github.com/abseil/abseil-cpp.git
```

## Authors:
- *Palonskaya Alina, B01-405 DREC*
- *Makarskaya Alexandra, B01-401 DREC*