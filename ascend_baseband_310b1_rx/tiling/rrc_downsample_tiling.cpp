#include <cassert>
#include <iostream>
#include <cstring>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace matmul_tiling;

extern "C" void GenerateTiling(const char *socVersion, uint8_t *tilingBuf)
{
    constexpr int32_t TILE_M = 32;
    constexpr int32_t N = 256;
    constexpr int32_t K = 1088;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    MultiCoreMatmulTiling tilingApi(*platform);

    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
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
    if (res == -1) { std::cerr << "Tiling failed!" << std::endl; return; }

    uint32_t sz = tilingData.GetDataSize();
    tilingData.SaveToBuffer(tilingBuf, sz);
    
}
