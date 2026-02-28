################################################################################
# smc2/CMakeLists.txt — SMC² RBPF Parameter Learner
#
# Library:
#   smc2_rbpf — SMC² + RBPF + Kalman tracker + CPMMH rejuvenation
#
# Tests:
#   test_smc2_rbpf            — SMC² convergence (8-param recovery)
#   test_param_tracker        — Kalman parameter tracker (Oracle vs Tracker)
#   test_4param_drift         — Dual-timescale theory validation
#   test_dsmc2_phased         — dSMC² pipeline (phased learning, 12-segment gauntlet)
#   test_bidirectional_phased — Bidirectional valve memory preservation
#   test_kalman_fair          — 2×2 factor analysis (traveling vs amnesiac × raw vs Kalman)
#   test_mu_learning          — μ learning diagnostic
#
# Dependencies from parent:
#   gpu_bpf_ptx_full  — PTX BPF library (for pipeline tests)
#   ptx_cubin_full    — cubin build target (runtime dependency)
################################################################################

# ==============================================================================
# Library: smc2_rbpf
# ==============================================================================

add_library(smc2_rbpf STATIC
    smc2_rbpf_batch.cu
    smc2_param_tracker.cu
    cpmmh_gpu_learn_v3.cu
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

set_target_properties(smc2_rbpf PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# ==============================================================================
# Helper: create an SMC² test executable with common settings
#
# Usage:
#   smc2_add_test(test_name [NEEDS_BPF] [EXTRA_SOURCES src1.cu ...])
# ==============================================================================

function(smc2_add_test TARGET_NAME)
    cmake_parse_arguments(ARG "NEEDS_BPF" "" "EXTRA_SOURCES" ${ARGN})

    add_executable(${TARGET_NAME} ${TARGET_NAME}.cu ${ARG_EXTRA_SOURCES})

    # All SMC² tests link against the library + CUDA runtime
    target_link_libraries(${TARGET_NAME} PRIVATE
        smc2_rbpf
        CUDA::cudart
        CUDA::curand
    )

    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Tests that use the production BPF pipeline
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
# Tests: pure SMC² (no BPF dependency)
# ==============================================================================

smc2_add_test(test_smc2_rbpf)
smc2_add_test(test_param_tracker)
smc2_add_test(test_4param_drift)

# ==============================================================================
# Tests: SMC² + BPF pipeline
# ==============================================================================

smc2_add_test(test_dsmc2_phased         NEEDS_BPF)
smc2_add_test(test_bidirectional_phased  NEEDS_BPF)
smc2_add_test(test_kalman_fair           NEEDS_BPF)
smc2_add_test(test_mu_learning           NEEDS_BPF)
