#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

WORLD_SIZE=2
M=16
K=128
N=128
TOPK=2
EXPERTS=16
MAX_OUTPUT_SIZE=32
AIC_NUM=28
AIV_NUM=56
SEED=20260515
ATOL=1e-4
RTOL=1e-3
GOLDEN_BACKEND=${DISPATCH_MEGA_COMBINE_GOLDEN_BACKEND:-python-batch}
GOLDEN_CHUNK_ROWS=${DISPATCH_MEGA_COMBINE_GOLDEN_CHUNK_ROWS:-512}
REUSE_DATA=${DISPATCH_MEGA_COMBINE_REUSE_DATA:-0}
START_SYNC=${DISPATCH_MEGA_COMBINE_START_SYNC:-0}
BUILD_ONLY=${BUILD_ONLY:-0}
export HCCL_WHITELIST_DISABLE=1

: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH must be set before running run.sh}"
CMAKE_COMPILER=${CMAKE_COMPILER:-bisheng}
MPI_ENV_BIN=${MPI_ENV_BIN:-/home/ntlab/miniconda3/envs/ltr_pto/bin}
MPI_ENV_LIB=${MPI_ENV_LIB:-/home/ntlab/miniconda3/envs/ltr_pto/lib}
MPI_LIB_PATH=${MPI_LIB_PATH:-${MPI_ENV_LIB}/libmpi.so}
MPI_RUNNER=${MPI_RUNNER:-mpirun}

export ASCEND_HOME_PATH
export PATH="${MPI_ENV_BIN}:$PATH"
export LD_LIBRARY_PATH="${MPI_ENV_LIB}:${LD_LIBRARY_PATH:-}"
export MPI_LIB_PATH

while [[ $# -gt 0 ]]; do
  case "$1" in
    --world-size) WORLD_SIZE="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --experts) EXPERTS="$2"; shift 2 ;;
    --max-output-size) MAX_OUTPUT_SIZE="$2"; shift 2 ;;
    --aic-num) AIC_NUM="$2"; shift 2 ;;
    --aiv-num) AIV_NUM="$2"; shift 2 ;;
    --atol) ATOL="$2"; shift 2 ;;
    --rtol) RTOL="$2"; shift 2 ;;
    --golden-backend) GOLDEN_BACKEND="$2"; shift 2 ;;
    --golden-chunk-rows) GOLDEN_CHUNK_ROWS="$2"; shift 2 ;;
    --reuse-data) REUSE_DATA=1; shift ;;
    --build-only) BUILD_ONLY=1; shift ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

if [[ "${AIV_NUM}" -ne $((AIC_NUM * 2)) ]]; then
  echo "dispatch_mega_combine expects a 1:2 mixed-core shape: aiv-num must equal aic-num*2" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_DIR="${SCRIPT_DIR}/out"
BUILD_DIR="${SCRIPT_DIR}/build"

GEN_DATA_EXTRA_ARGS=()
GEN_DATA_EXTRA_ARGS+=(--golden-backend "${GOLDEN_BACKEND}")
GEN_DATA_EXTRA_ARGS+=(--golden-chunk-rows "${GOLDEN_CHUNK_ROWS}")
if [[ "${REUSE_DATA}" != "0" ]]; then
  GEN_DATA_EXTRA_ARGS+=(--reuse-data)
fi

MIB=$((1024 * 1024))
HCCL_WINDOW_HEAD_GUARD_BYTES=4096
PACKED_OFFSET_A_BYTES=$((MAX_OUTPUT_SIZE * (K + 32)))
OFFSET_A_WINDOW_BYTES=$((PACKED_OFFSET_A_BYTES * 3))
OFFSET_D_BYTES=$((MAX_OUTPUT_SIZE * K * 2))
OFFSET_D_WINDOW_BYTES=$((((OFFSET_D_BYTES + 3 * MIB + 511) * 3 + 1) / 2))
NEEDED_WINDOW_BYTES="${OFFSET_A_WINDOW_BYTES}"
if [[ "${OFFSET_D_WINDOW_BYTES}" -gt "${NEEDED_WINDOW_BYTES}" ]]; then
  NEEDED_WINDOW_BYTES="${OFFSET_D_WINDOW_BYTES}"
fi
NEEDED_HCCL_BUFFSIZE_MB=$(((NEEDED_WINDOW_BYTES + HCCL_WINDOW_HEAD_GUARD_BYTES + MIB - 1) / MIB + 64))
CURRENT_HCCL_BUFFSIZE_MB="${HCCL_BUFFSIZE:-200}"
if [[ "${CURRENT_HCCL_BUFFSIZE_MB}" -lt "${NEEDED_HCCL_BUFFSIZE_MB}" ]]; then
  echo "[INFO] Raising HCCL_BUFFSIZE from ${CURRENT_HCCL_BUFFSIZE_MB} to ${NEEDED_HCCL_BUFFSIZE_MB} MB" \
    "for maxOutputSize=${MAX_OUTPUT_SIZE} K=${K} hcclHeadGuard=${HCCL_WINDOW_HEAD_GUARD_BYTES}"
  export HCCL_BUFFSIZE="${NEEDED_HCCL_BUFFSIZE_MB}"
fi

python3 "${SCRIPT_DIR}/scripts/gen_data.py" \
  --output-dir "${OUT_DIR}" \
  --world-size "${WORLD_SIZE}" \
  --m "${M}" --k "${K}" --n "${N}" \
  --topk "${TOPK}" --experts "${EXPERTS}" \
  --max-output-size "${MAX_OUTPUT_SIZE}" \
  --seed "${SEED}" \
  --atol "${ATOL}" \
  --rtol "${RTOL}" \
  "${GEN_DATA_EXTRA_ARGS[@]}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_COMPILER="${CMAKE_COMPILER}" \
  -DCMAKE_C_COMPILER="${CMAKE_COMPILER}" \
  -DCMAKE_CXX_COMPILER="${CMAKE_COMPILER}"
cmake --build "${BUILD_DIR}" --target dispatch_mega_combine -j16

if [[ "${BUILD_ONLY}" != "0" ]]; then
  echo "[INFO] BUILD_ONLY set; skipping mpirun execution (compile-only verification)."
  exit 0
fi

export LD_LIBRARY_PATH="${BUILD_DIR}/lib:${LD_LIBRARY_PATH}"
export DISPATCH_MEGA_COMBINE_CASE_DIR="${OUT_DIR}"
export DISPATCH_MEGA_COMBINE_AIC_NUM="${AIC_NUM}"
export DISPATCH_MEGA_COMBINE_AIV_NUM="${AIV_NUM}"
export DISPATCH_MEGA_COMBINE_START_SYNC="${START_SYNC}"
"${MPI_RUNNER}" -n "${WORLD_SIZE}" "${BUILD_DIR}/dispatch_mega_combine"
