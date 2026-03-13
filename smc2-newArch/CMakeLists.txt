################################################################################
# smc2-newArch/CMakeLists.txt — SMC² RBPF Parameter Learner (restructured)
#
# Library:
#   smc2_rbpf — SMC² engine + Kalman param tracker (with phased learning)
#
# Files:
#   smc2_engine.cu           All kernels + host logic
#   smc2_param_tracker.cu    Kalman tracker + phased learning + convergence
#
# Headers (public):
#   smc2_engine.cuh          Types, device utils, noise precision, sorting, API
#   smc2_param_tracker.cuh   Tracker types + API
#   smc2_pipeline.h          Convergence diagnostics + BPF pipeline wiring
#
# Host compiler: Intel ICX (icpx) from oneAPI
# Device target: sm_120 (Blackwell — RTX 5080)
#
# Tests: smc2-newArch/tests/
################################################################################

# ==============================================================================
# Host compiler: Intel ICX
#
# ICX advantages for this codebase:
#   - AVX-512 codegen tuned for host-side 8×8 Kalman/Cholesky math
#   - -fp-model=precise for reproducible Cholesky without perf loss on ICX
#   - -qopt-zmm-usage=high (no downclocking penalty on ICX/SPR silicon)
#
# Set before project() if you want CMake to pick it up for compiler detection.
# Alternatively, invoke cmake with:
#   cmake -B build -DCMAKE_CUDA_HOST_COMPILER=icpx
# ==============================================================================

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

# ── CUDA architecture ────────────────────────────────────────────────────────
# SM_120 = Blackwell (RTX 5080, B200).
# Override at cmake time: -DCMAKE_CUDA_ARCHITECTURES=90
set_target_properties(smc2_rbpf PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES         "120"
)

# ── NVCC flags ────────────────────────────────────────────────────────────────
target_compile_options(smc2_rbpf PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        --threads=0
        -lineinfo
        $<$<CONFIG:Release>:-O3 -DNDEBUG>
        $<$<CONFIG:Debug>:-O0 -G -g -DSMC2_DEBUG>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g -lineinfo>
    >
)

# ── Host compiler flags (passed through -Xcompiler) ──────────────────────────
#
# Three-way detection:
#   1. MSVC (cl.exe) — Windows default. Uses /W4, /O2, /arch:AVX512, /fp:precise
#   2. Intel ICX (icpx) — Best for AVX-512 host code. Uses -march, -qopt-zmm-usage
#   3. GCC/Clang — Linux/WSL fallback. Uses -march, -mavx512f etc.
#
# The detection uses CMAKE_CXX_COMPILER_ID which CMake resolves during
# project() — this is the compiler nvcc delegates host compilation to.

if(MSVC)
    # ── MSVC (cl.exe) ────────────────────────────────────────────────────
    message(STATUS "[smc2] MSVC detected as CUDA host compiler")
    set(SMC2_ICX_HOST FALSE)

    set(SMC2_HOST_RELEASE_FLAGS
        /W4
        /O2
        /fp:precise
        /arch:AVX512
    )
    set(SMC2_HOST_DEBUG_FLAGS
        /W4
        /Od
        /Zi
    )

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # ── Intel ICX (icpx) ─────────────────────────────────────────────────
    message(STATUS "[smc2] ICX detected as CUDA host compiler")
    set(SMC2_ICX_HOST TRUE)

    set(SMC2_HOST_RELEASE_FLAGS
        -fno-strict-aliasing
        -Wall -Wextra -Wno-unused-parameter
        -march=icelake-server
        -qopt-zmm-usage=high
        -fp-model=precise
        -O3 -fomit-frame-pointer
    )
    set(SMC2_HOST_DEBUG_FLAGS
        -fno-strict-aliasing
        -Wall -Wextra -Wno-unused-parameter
        -O0 -g -fno-omit-frame-pointer
    )

else()
    # ── GCC / Clang ──────────────────────────────────────────────────────
    message(STATUS "[smc2] ${CMAKE_CXX_COMPILER_ID} detected as CUDA host compiler")
    set(SMC2_ICX_HOST FALSE)

    set(SMC2_HOST_RELEASE_FLAGS
        -fno-strict-aliasing
        -Wall -Wextra -Wno-unused-parameter
        -march=icelake-server
        -mavx512f -mavx512bw -mavx512dq -mavx512vl
        -O3 -fomit-frame-pointer
    )
    set(SMC2_HOST_DEBUG_FLAGS
        -fno-strict-aliasing
        -Wall -Wextra -Wno-unused-parameter
        -O0 -g -fno-omit-frame-pointer
    )
endif()

# Convert list to comma-separated for -Xcompiler
list(JOIN SMC2_HOST_RELEASE_FLAGS "," _release_joined)
list(JOIN SMC2_HOST_DEBUG_FLAGS   "," _debug_joined)

target_compile_options(smc2_rbpf PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Release>:-Xcompiler=${_release_joined}>
        $<$<CONFIG:RelWithDebInfo>:-Xcompiler=${_release_joined}>
        $<$<CONFIG:Debug>:-Xcompiler=${_debug_joined}>
    >
)

# ── Optional feature flags ────────────────────────────────────────────────────
# Usage:
#   cmake -B build -DSMC2_NOISE_FP16=ON     # FP16 noise buffers
#   cmake -B build -DSMC2_USE_CUB_SORT=ON   # CUB BlockRadixSort

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

# ── Status summary ────────────────────────────────────────────────────────────
message(STATUS "[smc2] GPU arch: sm_120 (Blackwell)")
message(STATUS "[smc2] Host compiler: ${CMAKE_CXX_COMPILER_ID} (ICX=${SMC2_ICX_HOST})")
message(STATUS "[smc2] Sources: smc2_engine.cu, smc2_param_tracker.cu")

# ==============================================================================
# Tests
# ==============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()