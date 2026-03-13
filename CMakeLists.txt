################################################################################
# CMakeLists.txt — BPF PTX Project
#
# Directory layout:
#   gpu_bpf/              — PTX BPF library + PTX assembly + BPF tests
#   smc2/                 — SMC² RBPF parameter learner + SMC² tests
#   csv_bank/             — Data files
#
# Cross-platform: auto-detects MSVC (Windows) vs ICX/GCC (Linux)
# Device compiler: NVCC (CUDA Toolkit 13.x+)
# Target GPU:     SM_120 (Blackwell / RTX 5080)
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

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")

# ==============================================================================
# Platform-specific compiler flags
# ==============================================================================

if(MSVC)
    message(STATUS "Detected MSVC host compiler")
    set(CMAKE_C_FLAGS_RELEASE   "/W4 /O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/W4 /O2 /DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG     "/W4 /Od /Zi")
    set(CMAKE_CXX_FLAGS_DEBUG   "/W4 /Od /Zi")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler=/W4,/O2")
    set(CMAKE_CUDA_FLAGS_DEBUG   "${CMAKE_CUDA_FLAGS_DEBUG} -Xcompiler=/W4,/Od,/Zi")

elseif(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
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
# Subdirectories
# ==============================================================================

add_subdirectory(gpu_bpf)
#add_subdirectory(smc2)
#add_subdirectory(smc2-newArch)

# ==============================================================================
# Print summary
# ==============================================================================

message(STATUS "-------------------------------------------")
message(STATUS "BPF PTX Project")
message(STATUS "  Platform:          ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Host C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  CUDA architectures:${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "  gpu_bpf/  — PTX BPF library + tests")
message(STATUS "  smc2/     — SMC² RBPF library + tests")
message(STATUS "-------------------------------------------")