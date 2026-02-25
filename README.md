# Proto

```bash
wget https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx.proto

wget https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx

protoc --cpp_out=frontend/parser/onnx/src onnx.proto

mv frontend/parser/onnx/src/onnx.pb.h frontend/parser/onnx/include/
```

# Compile

```bash
g++ -std=c++17 \
    -Ifrontend/parser/onnx/include \
    -Ifrontend/parser/onnx/src \
    frontend/parser/onnx/src/parser.cpp \
    frontend/parser/onnx/src/onnx.pb.cc \
    frontend/main.cpp \
    -o run \
    -I/opt/homebrew/include \
    -L/opt/homebrew/lib \
    -lprotobuf \
    -labsl_log_internal_check_op \
    -labsl_log_internal_message \
    -labsl_strings \
    -labsl_base \
    -labsl_throw_delegate \
    -labsl_raw_logging_internal \
    -pthread
```

# Run

```bash
./run mnist-8.onnx
```
