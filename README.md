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

Use clang-format:
```bash
find frontend common flags tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i
```

Use cpp-chack:
```bash
cppcheck --enable=all --inconclusive --suppress=missingIncludeSystem --error-exitcode=1 --quiet frontend common flags tests
```

To get absl:
```bash
git clone https://github.com/abseil/abseil-cpp.git
cd abseil-cpp
```