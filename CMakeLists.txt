################################################################################
# CMakeLists.txt — BPF PTX Project
#
# Directory layout:
#   gpu_bpf/          — PTX BPF library (delegated)
#   ptx_assembly/     — Hand-written PTX kernels and pre-compiled cubins
#   smc2/             — SMC² RBPF parameter learner (delegated)
#   tests/            — BPF test harnesses and diagnostics
#   csv_bank/         — Data files
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
# PTX compilation (ahead-of-time PTX -> cubin)
# ==============================================================================

find_program(PTXAS_EXECUTABLE ptxas
    HINTS ${CUDAToolkit_BIN_DIR} ENV CUDA_PATH
    PATH_SUFFIXES bin
)

if(PTXAS_EXECUTABLE)
    message(STATUS "Found ptxas: ${PTXAS_EXECUTABLE}")
else()
    message(WARNING "ptxas not found — cubin will not be pre-compiled.")
endif()

# --- Full PTX (14 kernels) ---
set(PTX_FULL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/ptx_assembly/bpf_kernels_full.ptx")
set(CUBIN_FULL_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/bpf_kernels_full.cubin")

if(PTXAS_EXECUTABLE AND EXISTS "${PTX_FULL_SOURCE}")
    add_custom_command(
        OUTPUT  "${CUBIN_FULL_OUTPUT}"
        COMMAND "${PTXAS_EXECUTABLE}" -arch=sm_120 -o "${CUBIN_FULL_OUTPUT}" "${PTX_FULL_SOURCE}" --verbose
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
# Subdirectories
# ==============================================================================

add_subdirectory(gpu_bpf)
add_subdirectory(smc2)

# ==============================================================================
# BPF tests (live in tests/)
# ==============================================================================

# Helper for BPF tests
function(bpf_add_test TARGET_NAME SOURCE)
    add_executable(${TARGET_NAME} ${SOURCE})
    target_link_libraries(${TARGET_NAME} PRIVATE
        gpu_bpf_ptx_full CUDA::cuda_driver CUDA::cudart CUDA::curand
    )
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/gpu_bpf
    )
    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    if(TARGET ptx_cubin_full)
        add_dependencies(${TARGET_NAME} ptx_cubin_full)
    endif()
endfunction()

bpf_add_test(test_bpf_ptx_full       tests/test_bpf_matched_dgp.cu)
bpf_add_test(test_bpf_ptx_stress     tests/test_bpf_ptx_stress.cu)
bpf_add_test(test_jump_bands         tests/test_jump_bands.cu)

# bench_particle_scaling needs USE_PTX_FULL define
bpf_add_test(bench_particle_scaling   tests/bench_particle_scaling.cu)
target_compile_definitions(bench_particle_scaling PRIVATE USE_PTX_FULL=1)

# Tests that also need smc2_rbpf
bpf_add_test(test_bpf_adaptive_stress tests/test_bpf_adaptive_stress.cu)
target_link_libraries(test_bpf_adaptive_stress PRIVATE smc2_rbpf)

bpf_add_test(test_jump_stress         tests/test_jump_stress.cu)
target_link_libraries(test_jump_stress PRIVATE smc2_rbpf)

bpf_add_test(bench_grad_overhead      tests/bench_grad_overhead.cu)
target_link_libraries(bench_grad_overhead PRIVATE smc2_rbpf)
target_include_directories(bench_grad_overhead PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ==============================================================================
# Print summary
# ==============================================================================

message(STATUS "-------------------------------------------")
message(STATUS "BPF PTX Project Configuration")
message(STATUS "  Platform:          ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Host C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  CUDA architectures:${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "  Libraries:")
message(STATUS "    gpu_bpf_ptx_full  — PTX BPF (14 kernels, via gpu_bpf/)")
message(STATUS "    smc2_rbpf         — SMC² RBPF (8-param, via smc2/)")
message(STATUS "-------------------------------------------")
