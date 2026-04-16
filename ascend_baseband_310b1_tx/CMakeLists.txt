cmake_minimum_required(VERSION 3.16.0)
project(ascend_baseband_tx_chain_build)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOC_VERSION "Ascend310B1" CACHE STRING "SoC type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "CANN path")
set(RUN_MODE "npu" CACHE STRING "run mode")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)

# ---- AscendC cmake ----
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake not found.")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# ---- Kernel static library (6 operators) ----
ascendc_library(kernels STATIC
    kernels/ldpc_encode.cpp
    kernels/qam64_modulation.cpp
    kernels/ofdm_ifft.cpp
    kernels/ifft_postproc.cpp
    kernels/up_sample_rrc.cpp)

# ---- PyBind11 ----
execute_process(COMMAND python3 -m pybind11 --includes OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYBIND11_INC)
string(REPLACE " " ";" PYBIND11_INC_LIST "${PYBIND11_INC}")
string(REPLACE "-I" "" PYBIND11_INC_LIST "${PYBIND11_INC_LIST}")
execute_process(COMMAND python3-config --extension-suffix OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYBIND11_SUFFIX)
execute_process(COMMAND python3 -c "import numpy; print(numpy.get_include())" OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NUMPY_INC)

# ---- Shared library (pybind + tiling) ----
add_library(pybind11_lib SHARED pybind11.cpp
    tiling/ldpc_encode_tiling.cpp
    tiling/ofdm_ifft_tiling.cpp
    tiling/up_sample_rrc_tiling.cpp)

# Symbol renaming for tiling functions
set_source_files_properties(tiling/ldpc_encode_tiling.cpp
    PROPERTIES COMPILE_DEFINITIONS "GenerateTiling=LdpcEncGenerateTiling")
set_source_files_properties(tiling/ofdm_ifft_tiling.cpp
    PROPERTIES COMPILE_DEFINITIONS "GenerateTiling=IfftGenerateTiling")
# up_sample_rrc_tiling.cpp already exports RrcUpGenerateTiling

target_include_directories(pybind11_lib PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PYBIND11_INC_LIST}
    ${NUMPY_INC}
    ${CMAKE_CURRENT_BINARY_DIR}/include/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/autogen)

target_link_libraries(pybind11_lib PRIVATE
    kernels ascendcl tiling_api register platform ascendalog dl)

target_compile_options(pybind11_lib PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1 -fPIC)

set_target_properties(pybind11_lib PROPERTIES
    OUTPUT_NAME ascend_baseband_tx_chain${PYBIND11_SUFFIX}
    PREFIX ""
    SUFFIX "")