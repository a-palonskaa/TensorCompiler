import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import numpy as np

def create_small_model_with_conv():
    input_shape = [1, 1, 4, 4]
    weight_shape = [1, 1, 2, 2]

    input_tensor = helper.make_tensor_value_info(
        "input1", onnx.TensorProto.FLOAT, input_shape
    )
    weight_tensor = helper.make_tensor_value_info(
        "conv_weight", onnx.TensorProto.FLOAT, weight_shape
    )
    bias_tensor = helper.make_tensor_value_info(
        "conv_bias", onnx.TensorProto.FLOAT, [1]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 1, 3, 3]
    )

    weight_data = np.ones(weight_shape, dtype=np.float32)
    bias_data = np.zeros([1], dtype=np.float32)
    weight_initializer = numpy_helper.from_array(weight_data, name="conv_weight")
    bias_initializer = numpy_helper.from_array(bias_data, name="conv_bias")

    conv_node = helper.make_node(
        "Conv",
        inputs=["input1", "conv_weight", "conv_bias"],
        outputs=["conv_out"],
        kernel_shape=[2, 2],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
        dilations=[1, 1],
        group=1,
        name="conv_node"
    )

    add_node = helper.make_node(
        "Add",
        inputs=["conv_out", "input1"],
        outputs=["output"],
        name="add_node"
    )

    graph = helper.make_graph(
        [conv_node, add_node],
        "conv_add_graph",
        [input_tensor, weight_tensor, bias_tensor],
        [output_tensor],
        initializer=[weight_initializer, bias_initializer]
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 14)],
    )

    onnx.checker.check_model(model)
    return model

if __name__ == "__main__":
    onnx.save(create_small_model_with_conv(), "./models/conv_add.onnx")
