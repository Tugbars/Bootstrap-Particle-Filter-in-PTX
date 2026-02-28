################################################################################
# smc2/CMakeLists.txt — SMC² RBPF Parameter Learner
#
# Library:
#   smc2_rbpf — SMC² + RBPF + Kalman tracker + CPMMH rejuvenation
#
# Tests delegated to smc2/tests/
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
# Tests
# ==============================================================================

add_subdirectory(tests)