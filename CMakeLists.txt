################################################################################
# CMakeLists.txt — BPF PTX Educational Project
#
# Host compiler:  Intel ICX (oneAPI)
# Device compiler: NVCC (CUDA Toolkit)
# Target GPU:     SM_100 (Blackwell / RTX 5080)
#
# Build:
#   source /opt/intel/oneapi/setvars.sh   # or wherever your oneAPI lives
#   cmake -B build -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
#   cmake --build build -j
#
# Run:
#   ./build/bpf_ptx_demo bpf_kernels.ptx
################################################################################

cmake_minimum_required(VERSION 3.24)  # 3.24+ for proper CUDA + custom host compiler

project(bpf_ptx
    VERSION 1.0
    LANGUAGES C CXX CUDA
)

# ==============================================================================
# Compiler requirements
# ==============================================================================

# Verify we're actually using ICX
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(WARNING
        "Expected Intel ICX (IntelLLVM) but got ${CMAKE_CXX_COMPILER_ID}.\n"
        "Pass -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx to cmake."
    )
endif()

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# ==============================================================================
# CUDA configuration
# ==============================================================================

# Tell NVCC to use ICX as the host compiler
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")

# Target SM_100 (Blackwell). Also generate PTX for forward compat.
set(CMAKE_CUDA_ARCHITECTURES "100")

# NVCC flags
#   -Xcompiler flags get forwarded to ICX
#   --expt-relaxed-constexpr: needed for thrust with non-MSVC host
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall,-Wextra,-O3,-march=native")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")

# ICX host flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")

# ==============================================================================
# Find packages
# ==============================================================================

find_package(CUDAToolkit REQUIRED)

# ==============================================================================
# PTX compilation (ahead-of-time PTX → cubin)
# ==============================================================================

# Find ptxas from the CUDA toolkit
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
                -arch=sm_100
                -o "${CUBIN_OUTPUT}"
                "${PTX_SOURCE}"
                --verbose
        DEPENDS "${PTX_SOURCE}"
        COMMENT "Compiling PTX → cubin (SM_100)"
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
    CUDA::cuda_driver     # libcuda — Driver API (cuModuleLoad, cuLaunchKernel)
    CUDA::cudart          # Runtime API (for thrust interop)
)

# thrust headers come from the CUDA toolkit
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
# Target: bpf_cuda_reference (original CUDA C version for comparison)
#
# Uncomment if you have gpu_bpf.cu + gpu_bpf.cuh alongside this project.
# ==============================================================================

# if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gpu_bpf.cu")
#     add_library(bpf_cuda_ref STATIC gpu_bpf.cu)
#     target_link_libraries(bpf_cuda_ref PRIVATE
#         CUDA::cudart
#     )
#     set_target_properties(bpf_cuda_ref PROPERTIES
#         CUDA_SEPARABLE_COMPILATION OFF
#     )
# endif()

# ==============================================================================
# Print summary
# ==============================================================================

message(STATUS "───────────────────────────────────────────")
message(STATUS "BPF PTX Project Configuration")
message(STATUS "  Host C compiler:   ${CMAKE_C_COMPILER}")
message(STATUS "  Host C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  CUDA compiler:     ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA host compiler:${CMAKE_CUDA_HOST_COMPILER}")
message(STATUS "  CUDA architectures:${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA toolkit:      ${CUDAToolkit_TARGET_DIR}")
message(STATUS "  ptxas:             ${PTXAS_EXECUTABLE}")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "───────────────────────────────────────────")
