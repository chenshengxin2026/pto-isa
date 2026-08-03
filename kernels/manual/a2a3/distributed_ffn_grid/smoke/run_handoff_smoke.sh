#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

# GridPipe 接力计数 producer-handoff smoke test (THANDOFF).  Each cell drives ONE
# window through two phases with two different producers -- TPUSH/TPOP<EAST>, then
# THANDOFF, then TPUSH/TPOP<SOUTH> -- and the host checks every popped tile by
# stamp.  --p1-pops < --p1-tiles leaves the ring undrained and exercises the relay
# branch; --p1-pops == --p1-tiles drains it and exercises the rebase branch.

: "${ASCEND_CANN_PATH:=$(ls -1d /usr/local/Ascend/cann-*/set_env.sh 2>/dev/null | sort -V | tail -1)}"
if [ -z "${ASCEND_CANN_PATH}" ]; then
    echo "[ERROR] Cannot find CANN set_env.sh.  Set ASCEND_CANN_PATH explicitly."
    exit 1
fi
source "${ASCEND_CANN_PATH}"

SHORT=r:,v:,d:
LONG=run-mode:,soc-version:,device-id:,grid-rows:,grid-cols:,p1-tiles:,p1-pops:,p2-tiles:,token-tile:,model-tile:,build-only
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

BUILD_ONLY=0
while :; do
    case "$1" in
        (-r | --run-mode)    RUN_MODE="$2"; shift 2;;
        (-v | --soc-version) SOC_VERSION="$2"; shift 2;;
        (-d | --device-id)   DEVICE_ID="$2"; shift 2;;
        (--grid-rows)        HANDOFF_ROWS="$2"; shift 2;;
        (--grid-cols)        HANDOFF_COLS="$2"; shift 2;;
        (--p1-tiles)         HANDOFF_P1_TILES="$2"; shift 2;;
        (--p1-pops)          HANDOFF_P1_POPS="$2"; shift 2;;
        (--p2-tiles)         HANDOFF_P2_TILES="$2"; shift 2;;
        (--token-tile)       HANDOFF_T="$2"; shift 2;;
        (--model-tile)       HANDOFF_W="$2"; shift 2;;
        (--build-only)       BUILD_ONLY=1; shift;;
        (--) shift; break;;
        (*) echo "[ERROR] Unexpected option: $1"; exit 1;;
    esac
done

: "${RUN_MODE:=npu}"
: "${SOC_VERSION:=Ascend910B1}"
: "${HANDOFF_ROWS:=3}"
: "${HANDOFF_COLS:=4}"
: "${HANDOFF_P1_TILES:=2}"
: "${HANDOFF_P1_POPS:=1}"
: "${HANDOFF_P2_TILES:=1}"
: "${HANDOFF_T:=16}"
: "${HANDOFF_W:=64}"
: "${DEVICE_ID:=${ASCEND_DEVICE_ID:-${DEVICE_ID:-0}}}"

if [[ ! "${SOC_VERSION}" =~ ^Ascend ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}"
    exit 1
fi

rm -rf /dev/shm/sem.hccl* 2>/dev/null
ipcrm -a 2>/dev/null

echo "=== GridPipe 接力计数 producer-handoff smoke ==="
echo "  RUN_MODE: ${RUN_MODE}  SOC_VERSION: ${SOC_VERSION}  DEVICE_ID: ${DEVICE_ID}"
echo "  Grid: ${HANDOFF_ROWS}x${HANDOFF_COLS}  P1: ${HANDOFF_P1_TILES} pushed / ${HANDOFF_P1_POPS} popped  P2: ${HANDOFF_P2_TILES}  Tile: ${HANDOFF_T}x${HANDOFF_W}"
echo "==============================================="

# CMakeLists.txt lives in the parent demo directory; build from there.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_DIR}"

rm -rf build
mkdir build
cd build

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}
set -euo pipefail

cmake -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} \
      -DHANDOFF_ROWS=${HANDOFF_ROWS} -DHANDOFF_COLS=${HANDOFF_COLS} \
      -DHANDOFF_P1_TILES=${HANDOFF_P1_TILES} -DHANDOFF_P1_POPS=${HANDOFF_P1_POPS} \
      -DHANDOFF_P2_TILES=${HANDOFF_P2_TILES} \
      -DHANDOFF_T=${HANDOFF_T} -DHANDOFF_W=${HANDOFF_W} \
      ..
make -j16 handoff_smoke

if [ "${BUILD_ONLY}" -eq 1 ]; then
    echo "[INFO] --build-only requested; skipping run."
    exit 0
fi

echo ""
echo "=== Running GridPipe 接力计数 producer-handoff smoke ==="
./handoff_smoke --device-id "${DEVICE_ID}"
