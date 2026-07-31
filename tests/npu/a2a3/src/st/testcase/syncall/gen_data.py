#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

# SYNCALL participants must equal the physical core count, which differs across chips
# (910B1: AIC 24 / AIV 48, 910B4: AIC 20 / AIV 40). run_st.py runs this script (in build/)
# before the test binary and cannot pass the chip in, so we detect the chip here, write
# build/syncall_core_cfg.txt (read by the binary via ../syncall_core_cfg.txt), and size
# the golden tensors from the same numbers.

import os
import subprocess

import numpy as np


def detect_aic_count():
    """Return the cube (AI Core) count of the local chip. Default to 910B1 full die (24)."""
    aic = 24
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10).stdout
        # Reduced-die parts (e.g. 910B4) expose 20 cubes; full-die 910B1/B2/B3 expose 24.
        if "910B4" in out:
            aic = 20
    except Exception:
        pass
    return aic


if __name__ == "__main__":
    aic = detect_aic_count()
    aiv = aic * 2

    with open("syncall_core_cfg.txt", "w") as cfg:
        cfg.write(f"{aic} {aiv}\n")

    case_lengths = {
        "SYNCALLTest.case_aiv_only_all_blocks": aiv,
        "SYNCALLTest.case_soft_aiv_only_all_blocks": aiv,
        "SYNCALLTest.case_mix_1_1_all_blocks": aic * 2,
        "SYNCALLTest.case_soft_mix_1_1_all_blocks": aic * 2,
        "SYNCALLTest.case_mix_1_2_all_blocks": aic * 3,
        "SYNCALLTest.case_soft_mix_1_2_all_blocks": aic * 3,
        "SYNCALLTest.case_hard_aic_only_all_blocks": aic,
        "SYNCALLTest.case_soft_aic_only_all_blocks": aic,
    }

    for case_name, length in case_lengths.items():
        golden = np.ones(length, dtype=np.int32)
        os.makedirs(case_name, exist_ok=True)
        golden.tofile(os.path.join(case_name, "golden.bin"))
