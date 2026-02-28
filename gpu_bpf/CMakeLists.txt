################################################################################
# gpu_bpf/CMakeLists.txt — PTX Bootstrap Particle Filter
#
# Library:
#   gpu_bpf_ptx_full — 14-kernel hand-written PTX BPF
#                      PCG32 PRNG, ICDF normals, cuRAND chi² pregeneration
#
# PTX assembly:
#   ptx_assembly/bpf_kernels_full.ptx  →  compiled to cubin at build time
#
# Tests delegated to gpu_bpf/tests/
################################################################################

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

set(PTX_FULL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/ptx_assembly/bpf_kernels_full.ptx")
set(CUBIN_FULL_OUTPUT "${CMAKE_BINARY_DIR}/bpf_kernels_full.cubin")

if(PTXAS_EXECUTABLE AND EXISTS "${PTX_FULL_SOURCE}")
    add_custom_command(
        OUTPUT  "${CUBIN_FULL_OUTPUT}"
        COMMAND "${PTXAS_EXECUTABLE}" -arch=sm_120
                -o "${CUBIN_FULL_OUTPUT}" "${PTX_FULL_SOURCE}" --verbose
        DEPENDS "${PTX_FULL_SOURCE}"
        COMMENT "Compiling bpf_kernels_full.ptx -> cubin (SM_120)"
        VERBATIM
    )
    add_custom_target(ptx_cubin_full ALL DEPENDS "${CUBIN_FULL_OUTPUT}")
endif()

if(EXISTS "${PTX_FULL_SOURCE}")
    configure_file("${PTX_FULL_SOURCE}" "${CMAKE_BINARY_DIR}/bpf_kernels_full.ptx" COPYONLY)
endif()

# ==============================================================================
# Library: gpu_bpf_ptx_full
# ==============================================================================

add_library(gpu_bpf_ptx_full STATIC
    gpu_bpf_ptx_full.cu
)

target_link_libraries(gpu_bpf_ptx_full
    PRIVATE CUDA::cuda_driver CUDA::cudart CUDA::curand
)

target_include_directories(gpu_bpf_ptx_full PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(gpu_bpf_ptx_full PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# ==============================================================================
# Tests
# ==============================================================================

add_subdirectory(tests)