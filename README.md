# Generate

```bash
rm -rf build
cmake -S . -B build -DCMAKE-COMPILE-OPTIONS=ON
cmake --build build
```

# Run

```bash
./build/run [name].onnx
```
name from `./models`
image of graph is located in `./images`

