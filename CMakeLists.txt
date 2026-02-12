################################################################################
# CMakeLists.txt — BPF PTX Educational Project
#
# Directory layout:
#   gpu_bpf/          — Host-side BPF/APF implementations + PTX driver
#   ptx_assembly/     — Hand-written PTX kernels and pre-compiled cubins
#   smc2/             — SMC² batch parameter learner (BPF inner filter)
#   tests/            — Test harnesses and diagnostics
#   csv_bank/         — Data files
#
# Cross-platform: auto-detects MSVC (Windows) vs ICX/GCC (Linux)
# Device compiler: NVCC (CUDA Toolkit 13.x+)
# Target GPU:     SM_120 (Blackwell / RTX 5080)
#
# Windows (MSVC):
#   cmake -B build
#   cmake --build build --config Release -j
#
# Linux (ICX):
#   source /opt/intel/oneapi/setvars.sh
#   cmake -B build -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
#   cmake --build build -j
#
# Linux (GCC):
#   cmake -B build
#   cmake --build build -j
################################################################################

cmake_minimum_required(VERSION 3.24)

project(bpf_ptx
    VERSION 1.0
    LANGUAGES C CXX CUDA
)

# ==============================================================================
# Standards
# ==============================================================================

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# ==============================================================================
# CUDA configuration
# ==============================================================================

set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
set(CMAKE_CUDA_ARCHITECTURES "120")

# Common NVCC flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")

# ==============================================================================
# Platform-specific compiler flags
# ==============================================================================

if(MSVC)
    # MSVC (cl.exe) — Windows
    message(STATUS "Detected MSVC host compiler")

    set(CMAKE_C_FLAGS_RELEASE   "/W4 /O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/W4 /O2 /DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG     "/W4 /Od /Zi")
    set(CMAKE_CXX_FLAGS_DEBUG   "/W4 /Od /Zi")

    # NVCC -> MSVC forwarding (Release)
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler=/W4,/O2")
    set(CMAKE_CUDA_FLAGS_DEBUG   "${CMAKE_CUDA_FLAGS_DEBUG} -Xcompiler=/W4,/Od,/Zi")

elseif(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    # Intel ICX (oneAPI)
    message(STATUS "Detected Intel ICX host compiler")

    if(WIN32)
        set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   /W4 /O2")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /O2")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/W4,/O2")
    else()
        set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Wall -Wextra -O3 -march=native")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall,-Wextra,-O3,-march=native")
    endif()

else()
    # GCC / Clang
    message(STATUS "Detected ${CMAKE_CXX_COMPILER_ID} host compiler")

    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Wall -Wextra -O3 -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall,-Wextra,-O3,-march=native")

endif()

# ==============================================================================
# Find packages
# ==============================================================================

find_package(CUDAToolkit REQUIRED)

# ==============================================================================
# PTX compilation (ahead-of-time PTX -> cubin)
# ==============================================================================

find_program(PTXAS_EXECUTABLE ptxas
    HINTS ${CUDAToolkit_BIN_DIR}
          ENV CUDA_PATH
    PATH_SUFFIXES bin
)

if(PTXAS_EXECUTABLE)
    message(STATUS "Found ptxas: ${PTXAS_EXECUTABLE}")
else()
    message(WARNING "ptxas not found — cubin will not be pre-compiled. "
                    "JIT compilation at runtime still works.")
endif()

# --- Weight-only PTX (7 kernels) ---
set(PTX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/ptx_assembly/bpf_kernels.ptx")
set(CUBIN_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/bpf_kernels.cubin")

if(PTXAS_EXECUTABLE AND EXISTS "${PTX_SOURCE}")
    add_custom_command(
        OUTPUT  "${CUBIN_OUTPUT}"
        COMMAND "${PTXAS_EXECUTABLE}"
                -arch=sm_120
                -o "${CUBIN_OUTPUT}"
                "${PTX_SOURCE}"
                --verbose
        DEPENDS "${PTX_SOURCE}"
        COMMENT "Compiling bpf_kernels.ptx -> cubin (SM_120)"
        VERBATIM
    )
    add_custom_target(ptx_cubin ALL DEPENDS "${CUBIN_OUTPUT}")
endif()

if(EXISTS "${PTX_SOURCE}")
    configure_file("${PTX_SOURCE}" "${CMAKE_CURRENT_BINARY_DIR}/bpf_kernels.ptx" COPYONLY)
endif()

# --- Full PTX (14 kernels) ---
set(PTX_FULL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/ptx_assembly/bpf_kernels_full.ptx")
set(CUBIN_FULL_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/bpf_kernels_full.cubin")

if(PTXAS_EXECUTABLE AND EXISTS "${PTX_FULL_SOURCE}")
    add_custom_command(
        OUTPUT  "${CUBIN_FULL_OUTPUT}"
        COMMAND "${PTXAS_EXECUTABLE}"
                -arch=sm_120
                -o "${CUBIN_FULL_OUTPUT}"
                "${PTX_FULL_SOURCE}"
                --verbose
        DEPENDS "${PTX_FULL_SOURCE}"
        COMMENT "Compiling bpf_kernels_full.ptx -> cubin (SM_120)"
        VERBATIM
    )
    add_custom_target(ptx_cubin_full ALL DEPENDS "${CUBIN_FULL_OUTPUT}")
endif()

if(EXISTS "${PTX_FULL_SOURCE}")
    configure_file("${PTX_FULL_SOURCE}" "${CMAKE_CURRENT_BINARY_DIR}/bpf_kernels_full.ptx" COPYONLY)
endif()

# ==============================================================================
# Library: gpu_bpf (BPF + APF — nvcc compiled kernels, legacy)
# ==============================================================================

add_library(gpu_bpf STATIC gpu_bpf/gpu_bpf.cu)

target_link_libraries(gpu_bpf PRIVATE
    CUDA::cudart
    CUDA::curand
)

target_include_directories(gpu_bpf PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_bpf
    ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(gpu_bpf PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# ==============================================================================
# Library: gpu_bpf_ptx_full (100% PTX BPF + cuRAND chi2 pregeneration)
#
# 13 BPF kernels in hand-written PTX. PCG32 PRNG, ICDF normals.
# cuRAND used only for bulk chi2 generation. APF still nvcc-compiled.
# ==============================================================================

add_library(gpu_bpf_ptx_full STATIC gpu_bpf/gpu_bpf_ptx_full.cu)

target_link_libraries(gpu_bpf_ptx_full PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::curand
)

target_include_directories(gpu_bpf_ptx_full PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_bpf
    ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(gpu_bpf_ptx_full PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# ==============================================================================
# Library: smc2_bpf (SMC² batch parameter learner with BPF inner filter)
#
# Learns θ = (ρ, σ_z, μ) for stochastic volatility model using SMC² with
# CPMMH rejuvenation. Student-t observation model matches production BPF.
# Uses CUB BlockRadixSort for CPMMH particle coupling.
# ==============================================================================

add_library(smc2_bpf STATIC smc2/smc2_bpf_batch.cu)

target_link_libraries(smc2_bpf PRIVATE
    CUDA::cudart
    CUDA::curand
)

target_include_directories(smc2_bpf PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/smc2
    ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(smc2_bpf PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# ==============================================================================
# Target: test_mix_bands (band-based mixture proposal validation)
# ==============================================================================

add_executable(test_mix_bands tests/test_mix_bands.cu)

target_link_libraries(test_mix_bands PRIVATE
    gpu_bpf
    CUDA::cudart
    CUDA::curand
)

set_target_properties(test_mix_bands PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)



# ==============================================================================
# Target: test_bpf_ptx_full (matched-DGP BPF accuracy and throughput)
# ==============================================================================

add_executable(test_bpf_ptx_full tests/test_bpf_matched_dgp.cu)

target_link_libraries(test_bpf_ptx_full PRIVATE
    gpu_bpf_ptx_full
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::curand
)

target_include_directories(test_bpf_ptx_full PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_bpf
)

set_target_properties(test_bpf_ptx_full PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

if(TARGET ptx_cubin_full)
    add_dependencies(test_bpf_ptx_full ptx_cubin_full)
endif()

# ==============================================================================
# Target: bench_particle_scaling (latency vs particle count)
# ==============================================================================

add_executable(bench_particle_scaling tests/bench_particle_scaling.cu)

target_compile_definitions(bench_particle_scaling PRIVATE USE_PTX_FULL=1)

target_link_libraries(bench_particle_scaling PRIVATE
    gpu_bpf_ptx_full
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::curand
)

target_include_directories(bench_particle_scaling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_bpf
)

set_target_properties(bench_particle_scaling PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

if(TARGET ptx_cubin_full)
    add_dependencies(bench_particle_scaling ptx_cubin_full)
endif()

# ==============================================================================
# Target: test_bpf_adaptive_stress (adaptive bands stress test — brutal DGPs)
# ==============================================================================

add_executable(test_bpf_adaptive_stress tests/test_bpf_adaptive_stress.cu)

target_link_libraries(test_bpf_adaptive_stress PRIVATE
    gpu_bpf
    CUDA::cudart
    CUDA::curand
)

set_target_properties(test_bpf_adaptive_stress PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# ==============================================================================
# Target: test_bpf_ptx_stress (adaptive bands stress test — PTX kernels)
# ==============================================================================

add_executable(test_bpf_ptx_stress tests/test_bpf_ptx_stress.cu)

target_link_libraries(test_bpf_ptx_stress PRIVATE
    gpu_bpf_ptx_full
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::curand
)

set_target_properties(test_bpf_ptx_stress PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

if(TARGET ptx_cubin_full)
    add_dependencies(test_bpf_ptx_stress ptx_cubin_full)
endif()

# ==============================================================================
# Target: test_cond_resample (Conditional resampling — GPU)
# ==============================================================================

add_executable(test_cond_resample tests/test_cond_resample.cu gpu_bpf/gpu_bpf.cu)

set_target_properties(test_cond_resample PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

target_include_directories(test_cond_resample PRIVATE gpu_bpf)
target_link_libraries(test_cond_resample curand)

# ==============================================================================
# Target: test_smc2_bpf (SMC² + BPF convergence tests)
#
# Simulates from known SV model, checks parameter recovery.
# Tests: single window, multi-window, high-vol stress, CPMMH acceptance,
#        ρ vs σ_z identifiability.
# ==============================================================================

add_executable(test_smc2_bpf smc2/test_smc2_bpf.cu)

target_link_libraries(test_smc2_bpf PRIVATE
    smc2_bpf
    CUDA::cudart
    CUDA::curand
)

target_include_directories(test_smc2_bpf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/smc2
)

set_target_properties(test_smc2_bpf PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# ==============================================================================
# Target: test_smc2_bpf (SMC² + BPF convergence tests)
#
# Simulates from known SV model, checks parameter recovery.
# Tests: single window, multi-window, high-vol stress, CPMMH acceptance,
#        ρ vs σ_z identifiability.
# ==============================================================================

add_executable(test_mu_learning smc2/test_mu_learning.cu)

target_link_libraries(test_mu_learning PRIVATE
    gpu_bpf_ptx_full
    smc2_bpf
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::curand
)

target_include_directories(test_mu_learning PRIVATE ${CMAKE_SOURCE_DIR}/gpu_bpf)


set_target_properties(test_mu_learning PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)


# ==============================================================================
# Print summary
# ==============================================================================

message(STATUS "-------------------------------------------")
message(STATUS "BPF PTX Project Configuration")
message(STATUS "  Platform:          ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Host C compiler:   ${CMAKE_C_COMPILER_ID} (${CMAKE_C_COMPILER})")
message(STATUS "  Host C++ compiler: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
message(STATUS "  CUDA compiler:     ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA host compiler:${CMAKE_CUDA_HOST_COMPILER}")
message(STATUS "  CUDA architectures:${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA toolkit:      ${CUDAToolkit_TARGET_DIR}")
message(STATUS "  ptxas:             ${PTXAS_EXECUTABLE}")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "-------------------------------------------")