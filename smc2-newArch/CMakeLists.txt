################################################################################
# smc2/CMakeLists.txt — SMC² RBPF Parameter Learner
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
# Tests delegated to smc2/tests/
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

# ── ICX host flags (passed through -Xcompiler) ───────────────────────────────
#
# Detection: CMAKE_CUDA_HOST_COMPILER or CMAKE_CXX_COMPILER may be icpx.
# We check for "Intel" in the compiler ID to set ICX-specific flags.
# If the host compiler is GCC/Clang, we still set AVX-512 flags but skip
# the ICX-only options like -qopt-zmm-usage and -fp-model.

# Collect host flags based on detected compiler
set(SMC2_HOST_FLAGS_COMMON
    -fno-strict-aliasing
    -Wall -Wextra -Wno-unused-parameter
)

# Check if host compiler is ICX
# CMAKE_CUDA_HOST_COMPILER is what nvcc actually uses
get_property(_cuda_host_compiler TARGET smc2_rbpf PROPERTY CUDA_HOST_COMPILER)
if(NOT _cuda_host_compiler)
    set(_cuda_host_compiler "${CMAKE_CUDA_HOST_COMPILER}")
endif()

# Probe for Intel compiler
set(SMC2_ICX_HOST FALSE)
if(_cuda_host_compiler)
    execute_process(
        COMMAND ${_cuda_host_compiler} --version
        OUTPUT_VARIABLE _host_version
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_host_version MATCHES "Intel")
        set(SMC2_ICX_HOST TRUE)
    endif()
endif()

# Also check CMAKE_CXX_COMPILER_ID as fallback
if(NOT SMC2_ICX_HOST AND CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(SMC2_ICX_HOST TRUE)
endif()

if(SMC2_ICX_HOST)
    message(STATUS "[smc2] ICX detected as CUDA host compiler")
    set(SMC2_HOST_ARCH_FLAGS
        -march=icelake-server
        -qopt-zmm-usage=high
        -fp-model=precise
    )
else()
    message(STATUS "[smc2] Non-ICX host compiler (${CMAKE_CXX_COMPILER_ID}) — using GCC/Clang AVX-512 flags")
    set(SMC2_HOST_ARCH_FLAGS
        -march=icelake-server
        -mavx512f -mavx512bw -mavx512dq -mavx512vl
    )
endif()

# Assemble -Xcompiler string
set(SMC2_HOST_RELEASE_FLAGS ${SMC2_HOST_FLAGS_COMMON} ${SMC2_HOST_ARCH_FLAGS} -O3 -fomit-frame-pointer)
set(SMC2_HOST_DEBUG_FLAGS   ${SMC2_HOST_FLAGS_COMMON} -O0 -g -fno-omit-frame-pointer)

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
message(STATUS "[smc2] ICX host: ${SMC2_ICX_HOST}")
message(STATUS "[smc2] Sources: smc2_engine.cu, smc2_param_tracker.cu")

# ==============================================================================
# Tests
# ==============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()
