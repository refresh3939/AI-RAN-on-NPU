/**
 * @file data_extract_mm_tiling.cpp — 数据提取Matmul tiling
 *
 * [1192, 256] × [256, 224] → [1192, 224]
 * half × half → float → half
 * 8核, TILE_M=32
 */
#include <iostream>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

using namespace matmul_tiling;

extern "C" void DataExtGenerateTiling(const char *socVersion, uint8_t *tilingBuf)
{
    constexpr int32_t M = 1192;
    constexpr int32_t N = 224;      // K_DATA_PAD
    constexpr int32_t K = 256;      // N_FFT

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    uint32_t tcubeTilingSize = sizeof(optiling::TCubeTiling);

    optiling::TCubeTiling tilingData;
    MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingApi.SetCType(TPosition::VECOUT, CubeFormat::ND, DataType::DT_FLOAT);

    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);

    // 8核, 每核 ~149行, 内部按TILE_M=32处理
    int32_t perCore = (M + 7) / 8;    // 149
    tilingApi.SetSingleShape(32, N, -1);
    tilingApi.SetDim(8);

    tilingApi.SetBias(false);
    tilingApi.SetBufferSpace(-1, -1, -1);

    if (tilingApi.GetTiling(tilingData) == -1) {
        std::cerr << "[错误] DataExtract tiling 生成失败" << std::endl;
        return;
    }

    tilingData.SaveToBuffer(tilingBuf, tcubeTilingSize);
    *reinterpret_cast<uint64_t *>(tilingBuf + tcubeTilingSize) = ubSize;
}
