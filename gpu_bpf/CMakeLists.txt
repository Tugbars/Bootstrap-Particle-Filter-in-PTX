################################################################################
# gpu_bpf/CMakeLists.txt — PTX Bootstrap Particle Filter
#
# Library:
#   gpu_bpf_ptx_full — 14-kernel hand-written PTX BPF
#                      PCG32 PRNG, ICDF normals, cuRAND chi² pregeneration
#
# Source files:
#   gpu_bpf_ptx_full.cu    — Host driver (cubin loader, kernel dispatch)
#   gpu_bpf_full.cuh       — Public header
#
# Also in directory (header-only / compiled by consumers):
#   bpf_jump_diffusion.cu  — Jump diffusion extension
#   bpf_jump_diffusion.cuh — Jump diffusion header
################################################################################

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