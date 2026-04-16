#include <cassert>
#include <iostream>
#include <cstring>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace matmul_tiling;

/**
 * ★ 改动 vs 旧版:
 *   1. GenerateTiling → RrcUpGenerateTiling (CMakeLists符号名对应)
 *   2. 在 offset=512 写入默认 totalRows=1192
 *      kernel从同一偏移读取, pybind通过set_total_rows动态更新
 */
extern "C" void RrcUpGenerateTiling(const char *socVersion, uint8_t *tilingBuf)
{
    constexpr int32_t TILE_M = 32;
    constexpr int32_t N = 272;
    constexpr int32_t K = 272;
    constexpr int32_t DEFAULT_TOTAL_ROWS = 1192;
    constexpr int32_t TOTAL_ROWS_OFFSET  = 512;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    MultiCoreMatmulTiling tilingApi(*platform);

    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, true);
    tilingApi.SetCType(TPosition::VECOUT, CubeFormat::ND, DataType::DT_FLOAT);

    tilingApi.SetOrgShape(TILE_M, N, K);
    tilingApi.SetShape(TILE_M, N, K);

    if (platform->GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        tilingApi.SetSingleShape(TILE_M, N, -1);
        tilingApi.SetDim(1);
    } else {
        tilingApi.SetDim(platform->GetCoreNumAiv());
    }

    tilingApi.SetBias(false);
    tilingApi.SetBufferSpace(-1, -1, -1);

    optiling::TCubeTiling tilingData;
    int64_t res = tilingApi.GetTiling(tilingData);
    if (res == -1) { std::cerr << "RrcUp tiling failed!" << std::endl; return; }

    uint32_t sz = tilingData.GetDataSize();
    tilingData.SaveToBuffer(tilingBuf, sz);

    // ★ 在固定偏移写入 totalRows, 与kernel TOTAL_ROWS_OFFSET一致
    *reinterpret_cast<int32_t*>(tilingBuf + TOTAL_ROWS_OFFSET) = DEFAULT_TOTAL_ROWS;

    std::cout << "RrcUpTiling: M=" << TILE_M << " N=" << N << " K=" << K
              << " tilingSize=" << sz
              << " totalRows=" << DEFAULT_TOTAL_ROWS
              << " @offset=" << TOTAL_ROWS_OFFSET << std::endl;
}
