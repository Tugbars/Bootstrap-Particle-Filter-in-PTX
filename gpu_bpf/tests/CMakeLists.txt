################################################################################
# gpu_bpf/tests/CMakeLists.txt — BPF test executables
#
# All tests link against gpu_bpf_ptx_full (from parent).
# Tests marked NEEDS_SMC2 additionally link smc2_rbpf.
#
# File layout:
#   gpu_bpf/tests/test_bpf_matched_dgp.cu
#   gpu_bpf/tests/test_bpf_ptx_stress.cu
#   gpu_bpf/tests/test_bpf_adaptive_stress.cu
#   gpu_bpf/tests/test_jump_bands.cu
#   gpu_bpf/tests/test_jump_stress.cu
#   gpu_bpf/tests/test_rng_diag.cu
#   gpu_bpf/tests/bench_particle_scaling.cu
#   gpu_bpf/tests/bench_grad_overhead.cu
################################################################################

# ==============================================================================
# Helper: create a BPF test executable
#
# Usage:
#   bpf_add_test(test_name [NEEDS_SMC2])
# ==============================================================================

function(bpf_add_test TARGET_NAME)
    cmake_parse_arguments(ARG "NEEDS_SMC2" "" "" ${ARGN})

    add_executable(${TARGET_NAME} ${TARGET_NAME}.cu)

    target_link_libraries(${TARGET_NAME} PRIVATE
        gpu_bpf_ptx_full
        CUDA::cuda_driver
        CUDA::cudart
        CUDA::curand
    )

    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/gpu_bpf
    )

    if(ARG_NEEDS_SMC2)
        target_link_libraries(${TARGET_NAME} PRIVATE smc2_rbpf)
        target_include_directories(${TARGET_NAME} PRIVATE
            ${CMAKE_SOURCE_DIR}/smc2
        )
    endif()

    if(TARGET ptx_cubin_full)
        add_dependencies(${TARGET_NAME} ptx_cubin_full)
    endif()

    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endfunction()

# ==============================================================================
# Pure BPF tests
# ==============================================================================

bpf_add_test(test_bpf_matched_dgp)
bpf_add_test(test_bpf_ptx_stress)
bpf_add_test(test_jump_bands)
bpf_add_test(test_rng_diag)

bpf_add_test(bench_particle_scaling)
target_compile_definitions(bench_particle_scaling PRIVATE USE_PTX_FULL=1)

# ==============================================================================
# BPF tests that also need SMC² (for param structures / pipeline)
# ==============================================================================

bpf_add_test(test_bpf_adaptive_stress  NEEDS_SMC2)
bpf_add_test(test_jump_stress          NEEDS_SMC2)
bpf_add_test(bench_grad_overhead       NEEDS_SMC2)
