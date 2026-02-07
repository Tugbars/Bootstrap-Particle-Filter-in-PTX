################################################################################
# CMakeLists.txt — BPF PTX Educational Project
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

    # NVCC → MSVC forwarding (Release)
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler=/W4,/O2")
    set(CMAKE_CUDA_FLAGS_DEBUG   "${CMAKE_CUDA_FLAGS_DEBUG} -Xcompiler=/W4,/Od,/Zi")

elseif(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    # Intel ICX (oneAPI)
    message(STATUS "Detected Intel ICX host compiler")

    if(WIN32)
        # ICX on Windows uses MSVC-compatible flags
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
# PTX compilation (ahead-of-time PTX → cubin)
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

set(PTX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/bpf_kernels.ptx")
set(CUBIN_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/bpf_kernels.cubin")

if(PTXAS_EXECUTABLE)
    add_custom_command(
        OUTPUT  "${CUBIN_OUTPUT}"
        COMMAND "${PTXAS_EXECUTABLE}"
                -arch=sm_120
                -o "${CUBIN_OUTPUT}"
                "${PTX_SOURCE}"
                --verbose
        DEPENDS "${PTX_SOURCE}"
        COMMENT "Compiling PTX -> cubin (SM_120)"
        VERBATIM
    )
    add_custom_target(ptx_cubin ALL DEPENDS "${CUBIN_OUTPUT}")
endif()

# Copy PTX source to build dir so the demo can find it
configure_file(
    "${PTX_SOURCE}"
    "${CMAKE_CURRENT_BINARY_DIR}/bpf_kernels.ptx"
    COPYONLY
)

# ==============================================================================
# Target: bpf_ptx_demo (host driver that loads PTX at runtime)
# ==============================================================================

add_executable(bpf_ptx_demo bpf_ptx_host.cu)

target_link_libraries(bpf_ptx_demo PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
)

target_include_directories(bpf_ptx_demo PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(bpf_ptx_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

if(PTXAS_EXECUTABLE)
    add_dependencies(bpf_ptx_demo ptx_cubin)
endif()

# ==============================================================================
# Library: gpu_bpf (BPF + APF + IMM runtime kernels)
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
# Target: test_bpf_matched_dgp (matched-DGP test suite — nvcc kernels)
# ==============================================================================

add_executable(test_bpf_matched_dgp test_bpf_matched_dgp.cu)

target_link_libraries(test_bpf_matched_dgp PRIVATE
    gpu_bpf
    CUDA::cudart
    CUDA::curand
)

set_target_properties(test_bpf_matched_dgp PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# ==============================================================================
# Library: gpu_bpf_ptx (HYBRID: PTX weight processing + cuRAND propagation)
#
# Drop-in replacement for gpu_bpf. BPF reductions/resample run from
# hand-written PTX loaded at runtime. APF/IMM use nvcc-compiled kernels.
# ==============================================================================

add_library(gpu_bpf_ptx STATIC gpu_bpf/gpu_bpf_ptx.cu)

target_link_libraries(gpu_bpf_ptx PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::curand
)

target_include_directories(gpu_bpf_ptx PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_bpf
    ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(gpu_bpf_ptx PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# ==============================================================================
# Target: test_bpf_ptx (matched-DGP test suite — PTX hybrid path)
#
# Same test harness, linked against gpu_bpf_ptx instead of gpu_bpf.
# Requires bpf_kernels.ptx in working directory at runtime.
# ==============================================================================

add_executable(test_bpf_ptx test_bpf_matched_dgp.cu)

target_link_libraries(test_bpf_ptx PRIVATE
    gpu_bpf_ptx
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::curand
)

set_target_properties(test_bpf_ptx PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Copy PTX to build dir so test_bpf_ptx can find it
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/bpf_kernels.ptx"
    "${CMAKE_CURRENT_BINARY_DIR}/bpf_kernels.ptx"
    COPYONLY
)

# ==============================================================================
# Library: gpu_bpf_ptx_full (100% PTX for BPF, cuRAND only for APF)
#
# All 13 BPF kernels in hand-written PTX. PCG32 PRNG, ICDF normals.
# Zero cuRAND in BPF path. APF/IMM still nvcc-compiled.
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
# Target: test_bpf_ptx_full (full PTX BPF + nvcc APF)
#
# Uses gpu_bpf_full.cuh (PCG32 RNG state) instead of gpu_bpf.cuh.
# Requires bpf_kernels_full.cubin or bpf_kernels_full.ptx in cwd.
# ==============================================================================

add_executable(test_bpf_ptx_full test_bpf_matched_dgp.cu)

target_compile_definitions(test_bpf_ptx_full PRIVATE USE_PTX_FULL=1)

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

# Copy full PTX/cubin to build dir
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bpf_kernels_full.ptx")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/bpf_kernels_full.ptx"
        "${CMAKE_CURRENT_BINARY_DIR}/bpf_kernels_full.ptx"
        COPYONLY
    )
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bpf_kernels_full.cubin")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/bpf_kernels_full.cubin"
        "${CMAKE_CURRENT_BINARY_DIR}/bpf_kernels_full.cubin"
        COPYONLY
    )
endif()

# ==============================================================================
# Print summary
# ==============================================================================

message(STATUS "───────────────────────────────────────────")
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
message(STATUS "───────────────────────────────────────────")