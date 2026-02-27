import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import numpy as np


def create_small_model():
    input_shape = [2, 2]

    const_add_value = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    const_mul_value = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)

    const_add = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const_add"],
        value=numpy_helper.from_array(const_add_value),
    )

    const_mul = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const_mul"],
        value=numpy_helper.from_array(const_mul_value),
    )

    nodes = [
        const_add,
        const_mul,
        helper.make_node(
            "Add",
            inputs=["input", "const_add"],
            outputs=["add_out"],
        ),
        helper.make_node(
            "Mul",
            inputs=["add_out", "const_mul"],
            outputs=["output"],
        ),
    ]

    input_tensor = helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, input_shape
    )

    output_tensor = helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, input_shape
    )

    graph = helper.make_graph(
        nodes,
        "small_graph",
        [input_tensor],
        [output_tensor],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 14)],
    )

    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    onnx.save(create_small_model(), "small_model.onnx")
