# Proto

```bash
wget https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx.proto

wget https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx
```

# Compile

```bash
cmake -S . -B build -DCMAKE-COMPILE-OPTIONS=ON
cmake --build build
```

# Run

```bash
./run mnist-8.onnx
```
