################################################################################
# smc2-newArch/tests/CMakeLists.txt — SMC² test targets
#
# Host compiler flags come from the root CMakeLists.txt.
# gpu_bpf target comes from add_subdirectory(gpu_bpf) in root.
################################################################################

# ── test_smc2: Smoke test (SMC² engine + param tracker) ──────────────────────

add_executable(test_smc2 test_smc2.cu)

target_link_libraries(test_smc2 PRIVATE
    smc2_rbpf
    CUDA::cudart
    CUDA::curand
)

target_include_directories(test_smc2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_target_properties(test_smc2 PROPERTIES
    CUDA_ARCHITECTURES "120"
)

target_compile_options(test_smc2 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
    >
)

add_test(NAME smc2_smoke COMMAND test_smc2)

# ── test_kalman_fair: Accuracy test (requires gpu_bpf) ───────────────────────

add_executable(test_kalman_fair test_kalman_fair.cu)

target_link_libraries(test_kalman_fair PRIVATE
    smc2_rbpf
    gpu_bpf_ptx_full
    CUDA::cudart
    CUDA::curand
)

target_include_directories(test_kalman_fair PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/gpu_bpf
)

set_target_properties(test_kalman_fair PROPERTIES
    CUDA_ARCHITECTURES "120"
)

target_compile_options(test_kalman_fair PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
    >
)

add_test(NAME smc2_kalman_fair COMMAND test_kalman_fair)