################################################################################
# smc2/tests/CMakeLists.txt — SMC² test targets
################################################################################

add_executable(test_smc2 test_smc2.cu)

target_link_libraries(test_smc2 PRIVATE
    smc2_rbpf
    CUDA::cudart
    CUDA::curand
)

target_include_directories(test_smc2 PRIVATE
    ${CMAKE_SOURCE_DIR}/smc2
)

set_target_properties(test_smc2 PROPERTIES
    CUDA_ARCHITECTURES "120"
)

# Inherit ICX host flags from parent
target_compile_options(test_smc2 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        $<$<CONFIG:Debug>:-O0 -G -g>
        $<$<CONFIG:Release>:-O2>
    >
)

# Register with CTest
add_test(NAME smc2_smoke COMMAND test_smc2)
