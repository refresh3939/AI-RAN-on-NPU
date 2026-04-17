#include <cassert>
#include <iostream>
#include <cstring>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace matmul_tiling;

extern "C" void GenerateTiling(const char *socVersion, uint8_t *tilingBuf)
{
    constexpr int32_t TILE_M = 32;
    constexpr int32_t N = 256;   // N_tile=144, 272=144+128
    constexpr int32_t K = 272;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    MatmulApiTiling tilingApi(*platform);
    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, true);
    tilingApi.SetCType(TPosition::VECOUT, CubeFormat::ND, DataType::DT_FLOAT);
    tilingApi.SetShape(TILE_M, N, K);
    tilingApi.SetOrgShape(TILE_M, N, K);
    tilingApi.SetBufferSpace(-1, -1, -1);
    optiling::TCubeTiling tilingData;
    int64_t res = tilingApi.GetTiling(tilingData);
    if (res == -1) { std::cerr << "Tiling failed!" << std::endl; return; }
    uint32_t sz = tilingData.GetDataSize();
    tilingData.SaveToBuffer(tilingBuf, sz);
    std::cout << "Tiling: M=" << TILE_M << " N=" << N << " K=" << K << " size=" << sz << std::endl;
}