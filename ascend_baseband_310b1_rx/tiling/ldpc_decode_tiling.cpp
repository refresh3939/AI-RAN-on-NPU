/**
 * @file matmul_custom_tiling.cpp
 *
 * 【改动说明】在原版基础上增加 Tiling 3 @offset 4096:
 *   decoded[:,:256] × G_left_inv → info [M,256] × [256,256] → [M,256]
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

using namespace matmul_tiling;
using namespace std;

#define TILING_OFFSET_2 2048
#define TILING_OFFSET_3 4096

extern "C" void GenerateTiling(const char *socVersion, uint8_t *tilingBuf)
{
    constexpr int32_t M = 256;
    constexpr int32_t N = 256;
    constexpr int32_t K = 512;
    constexpr int32_t SINGLECORE_M = 32; 
    
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    
    uint32_t tcubeTilingSize = sizeof(optiling::TCubeTiling);

    // =========================================================
    // Tiling 1: Bits × H^T → Syndrome
    // [256, 512] × [512, 256] → [256, 256]
    // =========================================================
    optiling::TCubeTiling tilingData;
    MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false);
    tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false);
    tilingApi.SetCType(TPosition::VECCALC, CubeFormat::ND, DataType::DT_INT32);

    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    tilingApi.SetSingleShape(SINGLECORE_M, 256, -1);
    
    int32_t mBlockNum = M / SINGLECORE_M;
    int32_t nBlockNum = N / 256;
    tilingApi.SetDim(mBlockNum * nBlockNum);
    
    tilingApi.SetBias(false);
    tilingApi.SetBufferSpace(-1, -1, -1);

    if (tilingApi.GetTiling(tilingData) == -1) {
        std::cerr << "[错误] Tiling 1 生成失败" << std::endl;
        return;
    }

    // =========================================================
    // Tiling 2: Syndrome × H → Votes
    // [256, 256] × [256, 512] → [256, 512]  (B转置)
    // =========================================================
    MultiCoreMatmulTiling api2(*ascendcPlatform);
    optiling::TCubeTiling data2;

    api2.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false);
    api2.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, true);
    api2.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_INT32);

    api2.SetOrgShape(256, 512, 256);
    api2.SetShape(256, 512, 256);
    api2.SetSingleShape(SINGLECORE_M, 512, -1);
    
    int32_t mBlockNum2 = M / SINGLECORE_M;
    int32_t nBlockNum2 = 512 / 512;
    api2.SetDim(mBlockNum2 * nBlockNum2);

    api2.SetBufferSpace(-1, -1, -1);

    if (api2.GetTiling(data2) == -1) {
        std::cerr << "[错误] Tiling 2 生成失败" << std::endl;
        return;
    }

    // =========================================================
    // Tiling 3: Decoded[:,:256] × G_left_inv → Info
    // [256, 256] × [256, 256] → [256, 256]
    // =========================================================
    MultiCoreMatmulTiling api3(*ascendcPlatform);
    optiling::TCubeTiling data3;

    api3.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false);
    api3.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false);
    api3.SetCType(TPosition::VECCALC, CubeFormat::ND, DataType::DT_INT32);

    api3.SetOrgShape(M, 256, 256);
    api3.SetShape(M, 256, 256);
    api3.SetSingleShape(SINGLECORE_M, 256, -1);
    
    int32_t mBlockNum3 = M / SINGLECORE_M;
    int32_t nBlockNum3 = 256 / 256;
    api3.SetDim(mBlockNum3 * nBlockNum3);

    api3.SetBias(false);
    api3.SetBufferSpace(-1, -1, -1);

    if (api3.GetTiling(data3) == -1) {
        std::cerr << "[错误] Tiling 3 生成失败" << std::endl;
        return;
    }

    // =========================================================
    // 保存 Tiling 数据到缓冲区
    // =========================================================
    tilingData.SaveToBuffer(tilingBuf, tcubeTilingSize);
    *reinterpret_cast<uint64_t *>(tilingBuf + tcubeTilingSize) = ubSize;
    
    data2.SaveToBuffer(tilingBuf + TILING_OFFSET_2, tcubeTilingSize);
    *reinterpret_cast<uint64_t *>(tilingBuf + TILING_OFFSET_2 + tcubeTilingSize) = ubSize;
    
    data3.SaveToBuffer(tilingBuf + TILING_OFFSET_3, tcubeTilingSize);
    *reinterpret_cast<uint64_t *>(tilingBuf + TILING_OFFSET_3 + tcubeTilingSize) = ubSize;
}
