#!/bin/bash
# run_pybind.sh — TX Chain 构建脚本
CURRENT_DIR=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
SOC_VERSION="Ascend310B1"
while [[ $# -gt 0 ]]; do case "$1" in -v|--soc-version) SOC_VERSION="$2"; shift 2;; *) shift;; esac; done

if [ -n "$ASCEND_INSTALL_PATH" ]; then _AP=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then _AP=$ASCEND_HOME_PATH
else _AP=${HOME}/Ascend/ascend-toolkit/latest; [ ! -d "$_AP" ] && _AP=/usr/local/Ascend/ascend-toolkit/latest; fi
source $_AP/bin/setenv.bash
echo "[INFO] SOC: ${SOC_VERSION}, CANN: ${_AP}"

set -e
pip3 install pybind11 2>/dev/null || true

# 生成TX所需矩阵 (IDFT matrix, RRC Toeplitz, LDPC G matrix)
if [ -f "./scripts/gen_tx_matrices.py" ]; then
    python3 ./scripts/gen_tx_matrices.py
fi
python ./scripts/gen_matrices.py

rm -rf build && mkdir -p build
cmake -B build -DSOC_VERSION=${SOC_VERSION} -DASCEND_CANN_PACKAGE_PATH=${_AP}
cmake --build build -j$(nproc)

echo ""
ls -la build/ascend_baseband_tx_chain*.so 2>/dev/null && echo "[INFO] TX Chain BUILD SUCCESS" || echo "[ERROR] BUILD FAILED"
cp build/ascend_baseband_tx_chain*.so . 2>/dev/null || true
