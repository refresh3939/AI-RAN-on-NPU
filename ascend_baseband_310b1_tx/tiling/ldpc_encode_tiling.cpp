/**
 * @file LDPCEnc_custom_tiling.cpp
 * @brief LDPC 编码器 Tiling 配置生成
 * 
 * 矩阵乘法配置: [M, K] × [K, N] → [M, N]
 * 其中 M=256, K=256, N=512
 */
#include <cstring>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

using namespace matmul_tiling;

/**
 * @brief 生成 LDPC 编码的 Tiling 配置
 */
extern "C" void GenerateTiling(const char *socVersion, uint8_t *tilingBuf)
{
    // =========================================================
    // 配置参数
    // =========================================================
    constexpr int32_t M = 256;   // 批大小
    constexpr int32_t K = 256;   // 信息位长度
    constexpr int32_t N = 512;   // 码字长度
    
    // 关键：限制单核处理的块大小，避免 UB 越界
    // UB 约 192KB，需要 matmulBuf(int32) + castBuf(int16) + outputBuf(int16) + maskBuf(int16)
    // 32 * 512 * (4 + 2 + 2 + 2) = 163840 字节 ≈ 160KB，安全
    constexpr int32_t SINGLECORE_M = 32;
    constexpr int32_t SINGLECORE_N = 512;
    
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    uint64_t ubSize;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    
    // =========================================================
    // 配置矩阵乘法 Tiling
    // [M, K] × [K, N] → [M, N]
    // =========================================================
    optiling::TCubeTiling tilingData;
    MultiCoreMatmulTiling api(*platform);
    
    // 输入 A: 信息比特 [M, K] int8
    api.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false);
    // 输入 B: 生成矩阵 G [K, N] int8
    api.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false);
    // 输出 C: 编码结果 [M, N] int32 (后续转 int16 并 mod 2)
    api.SetCType(TPosition::VECCALC, CubeFormat::ND, DataType::DT_INT32);
    
    api.SetOrgShape(M, N, K);
    api.SetShape(M, N, K);
    
    // 关键修复：设置单核形状和核数
    api.SetSingleShape(SINGLECORE_M, SINGLECORE_N, -1);
    api.SetDim(M / SINGLECORE_M);  // 256/32 = 8 核
    
    api.SetBias(false);
    api.SetBufferSpace(-1, -1, -1);
    
    if (api.GetTiling(tilingData) == -1) {
        return;
    }
    
    // =========================================================
    // 保存到缓冲区
    // =========================================================
    uint32_t tilingSize = sizeof(optiling::TCubeTiling);
    tilingData.SaveToBuffer(tilingBuf, tilingSize);
    *reinterpret_cast<uint64_t*>(tilingBuf + tilingSize) = ubSize;
}