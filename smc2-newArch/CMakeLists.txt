################################################################################
# smc2-newArch/CMakeLists.txt — SMC² RBPF Parameter Learner (restructured)
#
# Library:
#   smc2_rbpf — SMC² engine + Kalman param tracker (with phased learning)
#
# Host compiler flags are set by the ROOT CMakeLists.txt (MSVC/ICX/GCC).
# This file only adds CUDA device flags and feature options.
################################################################################

# ==============================================================================
# Library: smc2_rbpf
# ==============================================================================

add_library(smc2_rbpf STATIC
    smc2_engine.cu
    smc2_param_tracker.cu
)

target_link_libraries(smc2_rbpf PRIVATE
    CUDA::cudart
    CUDA::curand
)

target_include_directories(smc2_rbpf PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/gpu_bpf
    ${CUDAToolkit_INCLUDE_DIRS}
)

# ── CUDA device settings ─────────────────────────────────────────────────────
set_target_properties(smc2_rbpf PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES         "120"
)

# CUDA device flags only — host flags come from root CMakeLists.txt
target_compile_options(smc2_rbpf PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        --threads=0
        -lineinfo
        $<$<CONFIG:Debug>:-G -g>
    >
)

# ── Optional feature flags ────────────────────────────────────────────────────
option(SMC2_NOISE_FP16    "Use FP16 noise storage (halves bandwidth, ~0.1% error)" OFF)
option(SMC2_USE_CUB_SORT  "Use CUB BlockRadixSort instead of bitonic"              OFF)

if(SMC2_NOISE_FP16)
    target_compile_definitions(smc2_rbpf PUBLIC SMC2_NOISE_FP16)
    message(STATUS "[smc2] Noise precision: FP16")
else()
    message(STATUS "[smc2] Noise precision: FP32 (default)")
endif()

if(SMC2_USE_CUB_SORT)
    target_compile_definitions(smc2_rbpf PUBLIC SMC2_USE_CUB_SORT)
    message(STATUS "[smc2] Sort backend: CUB BlockRadixSort")
else()
    message(STATUS "[smc2] Sort backend: Bitonic (default)")
endif()

# ── Status ────────────────────────────────────────────────────────────────────
message(STATUS "[smc2] Sources: smc2_engine.cu, smc2_param_tracker.cu")

# ==============================================================================
# Tests
# ==============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()