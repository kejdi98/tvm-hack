# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tvm
from tvm import te
import numpy as np
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_infer_type
from tvm.contrib import graph_executor
from tvm.relay.testing.temp_op_attr import TempOpAttr

# We use llvm target for testing functionality. `llvm` points to an older Intel
# generation machine, that legalizes to a simple lowering. Therefore, the
# legalization is overwritten such that it can be skipped and we use the
# QNNCanonicalizeOps lowering for the testing.
def legalize_qnn_conv2d(attrs, inputs, types):
    return None


def get_ref_func(
    data,
    weight,
    bias,
    zero_x,
    zero_y,
    effective_scale,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    out_dtype,
):
    if isinstance(zero_x, (int, float)):
        zero_x = relay.const(zero_x, "int32")
    if isinstance(zero_y, (int, float)):
        zero_y = relay.const(zero_y, "int32")

    casted_data = relay.op.cast(data, "int32")
    casted_weight = relay.op.cast(weight, "int32")
    shifted_data = relay.op.subtract(casted_data, zero_x)
    func = relay.op.nn.conv2d(
        data=shifted_data,
        weight=casted_weight,
        padding=padding,
        strides=strides,
        dilation=dilation,
        groups=groups,
        channels=channels,
        kernel_size=kernel_size,
        out_dtype=out_dtype,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
    )

    func = relay.Function(relay.analysis.free_vars(func), func)
    return func


def get_qnn_func(
    data,
    weight,
    bias,
    zero_x,
    zero_y,
    effective_scale,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    out_dtype,
):
    if isinstance(zero_x, int):
        zero_x = relay.const(zero_x, "int32")
    if isinstance(zero_y, int):
        zero_y = relay.const(zero_y, "int32")

    func = relay.op.nn.mcuconv2d(
        data,
        weight,
        bias,
        zero_x=zero_x,
        zero_y=zero_y,
        effective_scale=effective_scale,
        kernel_size=kernel_size,
        strides=strides,
        dilation=dilation,
        padding=padding,
        out_dtype=out_dtype,
        groups=groups,
        channels=channels,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
    )

    mod = relay.Function(relay.analysis.free_vars(func), func)
    mod = tvm.IRModule.from_expr(mod)
    print(mod)
    return mod


def get_funcs(
    data_shape,
    data_dtype,
    weight_shape,
    weight_dtype,
    bias_shape,
    bias_dtype,
    zero_x,
    zero_y,
    effective_scale,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    out_dtype,
):
    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=bias_dtype)

    ref_func = get_ref_func(
        data,
        weight,
        bias,
        zero_x,
        zero_y,
        effective_scale,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )

    ref_func = run_infer_type(ref_func)
    ref_func = tvm.IRModule.from_expr(ref_func)

    qnn_func = get_qnn_func(
        data,
        weight,
        bias,
        zero_x,
        zero_y,
        effective_scale,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )
    return (ref_func, qnn_func)


def verify(ref_func, qnn_func, data_shape, data_dtype, weight_shape, weight_dtype, bias_shape, bias_dtype):
    def get_inputs(data_shape, data_dtype, weight_shape, weight_dtype, bias_shape, bias_dtype):
        # Keeping inputs multiple of 4 because of a bug in Average Pool2d
        # https://discuss.tvm.apache.org/t/pool2d-gives-bad-output-for-integer-inputs/3377
        low_data = -128
        high_data = 127
        golden_data = np.random.randint(low=low_data, high=high_data, size=data_shape).astype(data_dtype)

        low_weight = -128
        high_weight = 127
        golden_weight = np.random.randint(low=low_weight, high=high_weight, size=weight_shape).astype(
            weight_dtype
        )

        if bias_dtype == "int32":
            low_bias = 10
            high_bias = 11
        golden_bias = np.random.randint(low=low_bias, high=high_bias, size=bias_shape).astype(bias_dtype)
        #golden_bias = np.array[10, 10, 10]
        return (golden_data, golden_weight, golden_bias)

    def get_output(func, golden_inputs):
        with tvm.transform.PassContext(opt_level=2):
            golden_data, golden_weight, golden_bias = golden_inputs
            params = {"weight": golden_weight, "bias": golden_bias}
            graph, lib, params = relay.build(func, "llvm", params=params)
            mod = graph_executor.create(graph, lib, device=tvm.cpu(0))
            mod.set_input("data", golden_data)
            mod.set_input(**params)
            mod.run()
            res = mod.get_output(0).numpy()
            return res

    golden_inputs = get_inputs(data_shape, data_dtype, weight_shape, weight_dtype, bias_shape, bias_dtype)
    golden_output = get_output(ref_func, golden_inputs)
    print(golden_output)
    qnn_output = get_output(qnn_func, golden_inputs)
    print(qnn_output)
    np.testing.assert_equal(qnn_output, golden_output)


def test_no_zero_point():
    with TempOpAttr("nn.mcuconv2d", "FTVMQnnLegalize", legalize_qnn_conv2d):

        # int8 input
        data_shape = (2, 1, 2, 4)
        data_dtype = "int8"
        weight_shape = (3, 1, 2, 2)
        weight_dtype = "int8"
        bias_shape = (3,)
        bias_dtype = "int32"
        effective_scales = [4, 3, 3]
        effective_scales = relay.const(np.array(effective_scales).astype("float32"))
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            weight_shape=weight_shape,
            weight_dtype=weight_dtype,
            bias_shape=bias_shape,
            bias_dtype=bias_dtype,
            zero_x=0,
            zero_y=0,
            effective_scale=effective_scales,
            strides=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            channels=3,
            kernel_size=(2, 2),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_layout="",
            out_dtype="int32",
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, weight_shape, weight_dtype, bias_shape, bias_dtype)

def test_per_channel_kernel_scale():
    with TempOpAttr("nn.mcuconv2d", "FTVMQnnLegalize", legalize_qnn_conv2d):
        data_shape = (2, 1, 2, 4)
        data_dtype = "int8"
        weight_shape = (3, 1, 2, 2)
        weight_dtype = "int8"
        bias_shape = (1, 1, 1, 1)
        bias_dtype = "int32"
        data = relay.var("data", shape=data_shape, dtype=data_dtype)
        weight = relay.var("kernel", shape=weight_shape, dtype=weight_dtype)
        bias = relay.var("bias", shape=bias_shape, dtype=bias_dtype)
        effective_scales = [2, 2, 2]
        effective_scales = relay.const(np.array(effective_scales).astype("float32"))

        func = relay.op.nn.mcuconv2d(
        data,
        weight,
        bias,
        zero_x=relay.const(0, "int32"),
        zero_y=relay.const(0, "int32"),
        effective_scale=effective_scales,
        kernel_size=(2, 2),
        strides=(1, 1),
        dilation=(1, 1),
        padding=(0, 0),
        out_dtype="int32",
        groups=1,
        channels=weight_shape[0],
        data_layout="NCHW",
        kernel_layout="int32",
    )

        mod = relay.Function(relay.analysis.free_vars(func), func)
        mod = tvm.IRModule.from_expr(mod)

if __name__ == "__main__":
    test_no_zero_point()
    test_per_channel_kernel_scale()
