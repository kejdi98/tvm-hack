import tvm
import numpy as np
from tvm import relay

def test_saturation():
    # Same params
    input_dtype = "int32"
    output_dtype = "int8"
    min = -128
    max = 127
    x = relay.var("x", shape=(1, 4), dtype=input_dtype)
    z = relay.op.nn.mcutruncate(
        x1=x,
        min=min,
        max=max,
        out_dtype=output_dtype,
    )

    func = relay.Function([x], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    #mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]
    mod = relay.transform.InferType()(mod)

    x_data = np.array((255, 255, 255, 255)).reshape((1, 4))
    golden_output = np.array((127, 127, 127, 127)).reshape((1, 4))

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
        x_data
    )
    np.testing.assert_equal(op_res.numpy(), golden_output)

if __name__ == "__main__":
    test_saturation()