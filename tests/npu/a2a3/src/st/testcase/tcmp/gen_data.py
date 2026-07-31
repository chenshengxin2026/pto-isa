#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import numpy as np
np.random.seed(19)

def gen_golden_data_tcmps(case_name, param):
    dtype = param.dtype

    H, W = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    if dtype in (np.float16, np.float32):
        dtype_info = np.finfo(dtype)
        input1 = np.random.uniform(low=-dtype_info.max, high=dtype_info.max, size=[H, W]).astype(dtype)
        input2 = np.random.uniform(low=-dtype_info.max, high=dtype_info.max, size=[H, W]).astype(dtype)
    else:
        dtype_info = np.iinfo(dtype)
        input1 = np.random.randint(dtype_info.min, dtype_info.max, size=[H, W]).astype(dtype)
        input2 = np.random.randint(dtype_info.min, dtype_info.max, size=[H, W]).astype(dtype)

    # Odd columns (0-indexed: 0, 2, 4, ...): same values
    input2[:, 0::2] = input1[:, 0::2]

    if getattr(param, "is_nan", False):
        input1[:] = np.nan
        input2[:] = np.nan

    if param.mode == "CmpMode::EQ":
        golden = np.isclose(input1, input2, rtol=0, atol=1e-9)
    if param.mode == "CmpMode::NE":
        golden = ~np.isclose(input1, input2, rtol=0, atol=1e-9)
    if param.mode == "CmpMode::LT":
        golden = (input1 < input2)
    if param.mode == "CmpMode::GT":
        golden = (input1 > input2)
    if param.mode == "CmpMode::GE":
        golden = (input1 >= input2)
    if param.mode == "CmpMode::LE":
        golden = (input1 <= input2)

    if getattr(param, "is_nan", False) and golden[0, 0] == 0:
        raise ValueError('Nan != Nan is not True, please check golden generation.')

    golden = golden.astype(np.uint8)
    golden[h_valid:, :] = 0
    golden[:, w_valid:] = 0

    bits_per_row = W // 8
    weights = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    out_uint8 = (golden.reshape(H, bits_per_row, 8) * weights).sum(axis=2, dtype=np.uint8).flatten()

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    np.array(out_uint8).astype(np.uint8).tofile("golden.bin")


class tcmpParams:
    def __init__(self, dtype, global_row, global_col, tile_row, tile_col, valid_row, valid_col, cmp_mode, is_nan=False):
        self.dtype = dtype
        self.global_row = global_row
        self.global_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.mode = cmp_mode
        self.is_nan = is_nan


def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
        np.int32: 'int32',
        np.int16: 'int16'
    }[param.dtype]
    cmpmode_str = {
        "CmpMode::EQ": 'eq',
        "CmpMode::NE": 'ne',
        "CmpMode::LT": 'lt',
        "CmpMode::GT": 'gt',
        "CmpMode::GE": 'ge',
        "CmpMode::LE": 'le'
    }[param.mode]
    nan_suffix = "_nan" if getattr(param, "is_nan", False) else ""
    return f"TCMPTest.case_{cmpmode_str}_{dtype_str}_{param.global_row}x{param.global_col}_"\
        f"{param.tile_row}x{param.tile_col}_{param.valid_row}x{param.valid_col}{nan_suffix}"


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        tcmpParams(np.float32, 1, 64, 1, 64, 1, 64, "CmpMode::EQ"),
        tcmpParams(np.float32, 8, 64, 8, 64, 8, 64, "CmpMode::GT"),
        tcmpParams(np.int32, 64, 64, 32, 32, 64, 64, "CmpMode::EQ"),
        tcmpParams(np.int32, 16, 32, 16, 32, 16, 32, "CmpMode::EQ"),
        tcmpParams(np.float32, 128, 128, 64, 64, 128, 128, "CmpMode::LE"),
        tcmpParams(np.int32, 77, 81, 32, 32, 77, 81, "CmpMode::EQ"),
        tcmpParams(np.int32, 32, 32, 32, 32, 32, 32, "CmpMode::EQ"),
        tcmpParams(np.float32, 32, 32, 32, 32, 32, 32, "CmpMode::NE"),
        tcmpParams(np.float32, 32, 32, 32, 32, 32, 32, "CmpMode::NE", is_nan=True),
        tcmpParams(np.float32, 2, 4096, 2, 4096, 2, 4096, "CmpMode::LT"),
        tcmpParams(np.float32, 2, 4096, 2, 4096, 2, 4096, "CmpMode::NE"),
        tcmpParams(np.float32, 2, 4096, 2, 4096, 2, 4096, "CmpMode::NE", is_nan=True),
        tcmpParams(np.float32, 1, 64, 1, 64, 1, 64, "CmpMode::NE"),
        tcmpParams(np.float32, 64, 64, 32, 32, 64, 64, "CmpMode::NE"),
        tcmpParams(np.float16, 1, 64, 1, 64, 1, 64, "CmpMode::EQ"),
        tcmpParams(np.float16, 8, 64, 8, 64, 8, 64, "CmpMode::NE"),
        tcmpParams(np.float16, 32, 32, 32, 32, 32, 32, "CmpMode::LT"),
        tcmpParams(np.float16, 32, 32, 32, 32, 32, 32, "CmpMode::GT"),
        tcmpParams(np.float16, 16, 32, 16, 32, 16, 32, "CmpMode::GE"),
        tcmpParams(np.float16, 128, 128, 64, 64, 128, 128, "CmpMode::LE"),
        tcmpParams(np.float16, 32, 32, 32, 32, 32, 32, "CmpMode::NE", is_nan=True),
        tcmpParams(np.float16, 2, 4096, 2, 4096, 2, 4096, "CmpMode::NE"),
        tcmpParams(np.float16, 128, 128, 64, 64, 128, 128, "CmpMode::NE"),
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tcmps(case_name, param)
        os.chdir(original_dir)
