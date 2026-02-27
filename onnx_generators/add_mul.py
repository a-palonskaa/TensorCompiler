import onnx
import onnx.helper as helper
import numpy as np

def create_small_model_no_constants():
    input_shape = [2, 2]

    input_tensor = helper.make_tensor_value_info(
        "input1", onnx.TensorProto.FLOAT, input_shape
    )
    const_add_tensor = helper.make_tensor_value_info(
        "input2", onnx.TensorProto.FLOAT, input_shape
    )
    const_mul_tensor = helper.make_tensor_value_info(
        "input3", onnx.TensorProto.FLOAT, input_shape
    )
    output_tensor = helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, input_shape
    )

    nodes = [
        helper.make_node(
            "Add",
            inputs=["input1", "input2"],
            outputs=["add_out"],
        ),
        helper.make_node(
            "Mul",
            inputs=["add_out", "input3"],
            outputs=["output"],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "add_mul",
        [input_tensor, const_add_tensor, const_mul_tensor],
        [output_tensor],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 14)],
    )

    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    onnx.save(create_small_model_no_constants(), "./models/add_mul.onnx")
