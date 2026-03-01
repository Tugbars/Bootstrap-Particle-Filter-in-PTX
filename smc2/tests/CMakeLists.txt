################################################################################
# smc2/tests/CMakeLists.txt — SMC² test executables
#
# All tests link against smc2_rbpf (from parent).
# Tests marked NEEDS_BPF additionally link gpu_bpf_ptx_full + cuda_driver.
#
# File layout after move:
#   smc2/tests/test_smc2_rbpf.cu
#   smc2/tests/test_param_tracker.cu
#   smc2/tests/test_4param_drift.cu
#   smc2/tests/test_dsmc2_phased.cu
#   smc2/tests/test_bidirectional_phased.cu
#   smc2/tests/test_kalman_fair.cu
#   smc2/tests/test_mu_learning.cu
#   smc2/tests/test_convergence_diag.cu
#   smc2/tests/test_convergence_gate.cu
################################################################################

# ==============================================================================
# Helper: create an SMC² test executable
#
# Usage:
#   smc2_add_test(test_name [NEEDS_BPF])
# ==============================================================================

function(smc2_add_test TARGET_NAME)
    cmake_parse_arguments(ARG "NEEDS_BPF" "" "" ${ARGN})

    add_executable(${TARGET_NAME} ${TARGET_NAME}.cu)

    target_link_libraries(${TARGET_NAME} PRIVATE
        smc2_rbpf
        CUDA::cudart
        CUDA::curand
    )

    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/smc2
    )

    if(ARG_NEEDS_BPF)
        target_link_libraries(${TARGET_NAME} PRIVATE
            gpu_bpf_ptx_full
            CUDA::cuda_driver
        )
        target_include_directories(${TARGET_NAME} PRIVATE
            ${CMAKE_SOURCE_DIR}/gpu_bpf
        )
        if(TARGET ptx_cubin_full)
            add_dependencies(${TARGET_NAME} ptx_cubin_full)
        endif()
    endif()

    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endfunction()

# ==============================================================================
# Pure SMC² tests (no BPF dependency)
# ==============================================================================

smc2_add_test(test_smc2_rbpf)
smc2_add_test(test_param_tracker)
smc2_add_test(test_4param_drift)
smc2_add_test(test_convergence_diag)
smc2_add_test(test_convergence_gate      NEEDS_BPF)

# ==============================================================================
# SMC² + BPF pipeline tests
# ==============================================================================

smc2_add_test(test_dsmc2_phased         NEEDS_BPF)
smc2_add_test(test_bidirectional_phased  NEEDS_BPF)
smc2_add_test(test_kalman_fair           NEEDS_BPF)
smc2_add_test(test_mu_learning           NEEDS_BPF)