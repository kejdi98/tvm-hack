import tvm
import numpy as np
from tvm import relay


def test_tflite_same_io_qnn_params():
    data_dtype = "uint8"
    scale_dtype = "float32"
    zero_dtype = "int32"
    x = relay.var("x", shape=(1, 4), dtype=data_dtype)
    y = relay.var("y", shape=(1, 4), dtype=data_dtype)
    scale_x = relay.const(0.00784314, "float32")
    scale_y = relay.const(0.00784314, "float32")
    scale_out = relay.const(0.00784314, "float32")
    zero_x = relay.const(127, "int32")
    zero_y = relay.const(127, "int32")
    zero_out = relay.const(127, "int32")
    out_dtype = "int8"
    z = relay.op.nn.mcuadd(
        x1=x,
        x2=y,
        zero_x1=zero_x,
        zero_x2=zero_y,
        scale_x1=scale_x,
        scale_x2=scale_y,
        zero_y=zero_out,
        scale_y=scale_out,
        out_dtype=out_dtype,
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    print(mod)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    print(mod)
    func = mod["main"]
    print(func)

    x_datas = [
        np.array((140, 153, 165, 178)).reshape((1, 4)),
        np.array((25, 153, 178, 216)).reshape((1, 4)),
        np.array((25, 153, 216, 165)).reshape((1, 4)),
    ]
    y_datas = [
        np.array((204, 178, 165, 140)).reshape((1, 4)),
        np.array((204, 178, 191, 25)).reshape((1, 4)),
        np.array((204, 178, 25, 191)).reshape((1, 4)),
    ]
    golden_outputs = [
        np.array((217, 204, 203, 191)).reshape((1, 4)),
        np.array((102, 204, 242, 114)).reshape((1, 4)),
        np.array((102, 204, 114, 229)).reshape((1, 4)),
    ]

    for i in range(0, 3):
        x_data = x_datas[i]
        y_data = y_datas[i]
        golden_output = golden_outputs[i]

        op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
            x_data, y_data
        )
        np.testing.assert_equal(op_res.numpy(), golden_output)


def test_saturation():
    # Same params
    data_dtype = "uint8"
    x = relay.var("x", shape=(1, 4), dtype=data_dtype)
    y = relay.var("y", shape=(1, 4), dtype=data_dtype)
    z = relay.op.nn.mcuadd(
        x1=x,
        x2=y,
        zero_x1=relay.const(0, "int32"),
        zero_x2=relay.const(0, "int32"),
        scale_x1=relay.const(0.125, "float32"),
        scale_x2=relay.const(0.125, "float32"),
        zero_y=relay.const(0, "int32"),
        scale_y=relay.const(0.125, "float32"),
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]
    mod = relay.transform.InferType()(mod)

    x_data = np.array((255, 1, 1, 0)).reshape((1, 4))
    y_data = np.array((255, 255, 128, 0)).reshape((1, 4))
    golden_output = np.array((255, 255, 129, 0)).reshape((1, 4))

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
        x_data, y_data
    )
    np.testing.assert_equal(op_res.numpy(), golden_output)

    # Same params, different scale
    z = relay.op.nn.mcuadd(
        x1=x,
        x2=y,
        zero_x1=relay.const(0, "int32"),
        zero_x2=relay.const(0, "int32"),
        scale_x1=relay.const(0.125, "float32"),
        scale_x2=relay.const(0.125, "float32"),
        zero_y=relay.const(0, "int32"),
        scale_y=relay.const(0.25, "float32"),
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_data = np.array((255, 1, 1, 0)).reshape((1, 4))
    y_data = np.array((255, 255, 127, 0)).reshape((1, 4))
    golden_output = np.array((255, 129, 65, 0)).reshape((1, 4))

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
        x_data, y_data
    )
    np.testing.assert_equal(op_res.numpy(), golden_output)

    # Same io params, different output scale
    z = relay.op.nn.mcuadd(
        x1=x,
        x2=y,
        zero_x1=relay.const(0, "int32"),
        zero_x2=relay.const(0, "int32"),
        scale_x1=relay.const(0.125, "float32"),
        scale_x2=relay.const(0.125, "float32"),
        zero_y=relay.const(0, "int32"),
        scale_y=relay.const(0.25, "float32"),
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_data = np.array((255, 1, 1, 0)).reshape((1, 4))
    y_data = np.array((255, 255, 127, 0)).reshape((1, 4))
    golden_output = np.array((255, 129, 65, 0)).reshape((1, 4))

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
        x_data, y_data
    )
    np.testing.assert_equal(op_res.numpy(), golden_output)

    # All params different
    z = relay.op.nn.mcuadd(
        x1=x,
        x2=y,
        zero_x1=relay.const(0, "int32"),
        zero_x2=relay.const(0, "int32"),
        scale_x1=relay.const(0.5, "float32"),
        scale_x2=relay.const(0.25, "float32"),
        zero_y=relay.const(0, "int32"),
        scale_y=relay.const(0.125, "float32"),
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_data = np.array((255, 0, 1, 0)).reshape((1, 4))
    y_data = np.array((0, 128, 64, 0)).reshape((1, 4))
    golden_output = np.array((255, 255, 132, 0)).reshape((1, 4))

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
        x_data, y_data
    )
    np.testing.assert_equal(op_res.numpy(), golden_output)


if __name__ == "__main__":
    test_tflite_same_io_qnn_params()
    test_saturation()