/**
 * @file matmul_LS_custom_tiling.cpp
 * @brief LS信道估计 + ZF均衡融合算子 Tiling配置 (v2: 1192行)
 * 
 * 关键变化:
 *   - M改为TILE_ROWS(32), 因为kernel内部循环, 每次matmul仍是32行
 *   - totalRows(1192)通过tiling尾部传入kernel
 *   - 8核并行, 每核处理~149行
 */
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <cstring>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace matmul_tiling;
using namespace std;

extern "C" void GenerateTiling(const char *socVersion, uint8_t *tilingBuf)
{
    // Matmul维度: 每次tile做 [32, 32] × [32, 512] → [32, 512]
    // kernel内部循环多次tile
    constexpr int32_t TILE_M = 32;          // 每次matmul的行数
    constexpr int32_t N = 512;              // LS矩阵输出维度
    constexpr int32_t K = 32;               // LS矩阵输入维度(导频)
    constexpr int32_t TOTAL_ROWS = 1192;    // 总OFDM符号数
    
    // 单核matmul配置 (每次tile处理TILE_M行)
    constexpr int32_t SINGLECORE_M = TILE_M;
    constexpr int32_t SINGLECORE_N = N;
    
    // 矩阵类型配置
    TPosition leftPosition = TPosition::GM;     
    CubeFormat leftFormat = CubeFormat::ND;
    DataType leftDtype = DataType::DT_FLOAT16;  
    bool isTransA = false;
    
    TPosition rightPosition = TPosition::GM;    
    CubeFormat rightFormat = CubeFormat::ND; 
    DataType rightDtype = DataType::DT_FLOAT16; 
    bool isTransB = false;
    
    TPosition resultPosition = TPosition::GM;   
    CubeFormat resultFormat = CubeFormat::ND;
    DataType resultDtype = DataType::DT_FLOAT;  
    
    bool isBias = false;
    
    optiling::TCubeTiling tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    
    MultiCoreMatmulTiling tilingApi(*ascendcPlatform);
    
    tilingApi.SetAType(leftPosition, leftFormat, leftDtype, isTransA);
    tilingApi.SetBType(rightPosition, rightFormat, rightDtype, isTransB);
    tilingApi.SetCType(resultPosition, resultFormat, resultDtype);
    
    // 单次tile的shape
    tilingApi.SetOrgShape(TILE_M, N, K);
    tilingApi.SetShape(TILE_M, N, K);
    
    if (ascendcPlatform->GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        tilingApi.SetSingleShape(SINGLECORE_M, SINGLECORE_N, -1);
        // 单核执行matmul (kernel内部每个core独立调matmul)
        tilingApi.SetDim(1);
    } else {
        tilingApi.SetDim(ascendcPlatform->GetCoreNumAiv());
    }
    
    tilingApi.SetBias(isBias);
    tilingApi.SetBufferSpace(-1, -1, -1);
    
    int64_t res = tilingApi.GetTiling(tilingData);
    if (res == -1) {
        std::cout << "ERROR: gen tiling failed!" << std::endl;
        return;
    }
    
    uint32_t tcubeTilingSize = tilingData.GetDataSize();
    tilingData.SaveToBuffer(tilingBuf, tcubeTilingSize);
    
    // 追加参数
    uint32_t offset = tcubeTilingSize;
    
    // localMemSize
    uint64_t localMemSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);
    memcpy(tilingBuf + offset, &localMemSize, sizeof(uint64_t));
    offset += sizeof(uint64_t);
    
    // totalRows
    int32_t totalRows = TOTAL_ROWS;
    memcpy(tilingBuf + offset, &totalRows, sizeof(int32_t));
    offset += sizeof(int32_t);
}