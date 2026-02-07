# BPF Bootstrap Particle Filter — Hand-Written PTX (SM_120 / RTX 5080)

## Results

Stochastic Volatility tracking across 5000 ticks — true latent log-volatility (white) vs BPF filtered estimate (blue). All scenarios use zero model mismatch (filter knows the true DGP parameters).

<img width="1907" height="947" alt="image" src="https://github.com/user-attachments/assets/5528f997-065e-4018-a5d6-67f729a348b6" />

<img width="1908" height="947" alt="image" src="https://github.com/user-attachments/assets/c68c4be1-8cd7-40f6-a10a-39bd590f76d0" />

<img width="1908" height="947" alt="image" src="https://github.com/user-attachments/assets/af15aa53-5f50-4eed-8e3e-77c9b5a82e6e" />

<img width="1907" height="587" alt="image" src="https://github.com/user-attachments/assets/00e8d2b3-b583-4594-b4bc-69504ec75023" />

**Performance:** 50K particles, ~130 µs/tick on RTX 5080. Hand-written PTX runs 15% faster than nvcc-compiled CUDA C — `ex2.approx`, `rcp.approx`, no `cvta.to.global` overhead, tighter register allocation.

## What This Is

A complete Bootstrap Particle Filter for Stochastic Volatility estimation,
written in raw NVIDIA PTX assembly targeting **SM_120 (Blackwell / RTX 5080)**.

8 hand-written kernels implement the full BPF pipeline: propagate + weight,
log-weight normalization (reduce-max, exp-sub, reduce-sum, scale),
log-likelihood accumulation, and systematic resampling via binary search.

This is an educational companion to the CUDA C implementation (`gpu_bpf.cu`).
Every kernel that the CUDA compiler (`nvcc`) would generate is written here by hand,
so you can see exactly what happens at the instruction level.

## Files

| File | Description |
|------|-------------|
| `bpf_kernels.ptx` | All 8 BPF kernels in raw PTX |
| `bpf_ptx_host.cu` | Host driver using CUDA Driver API to load & launch PTX |

## Build & Run

```bash
# Option 1: Ahead-of-time compile PTX → cubin, then load at runtime
ptxas -arch=sm_120 -o bpf_kernels.cubin bpf_kernels.ptx
nvcc -o bpf_demo bpf_ptx_host.cu -lcuda -arch=sm_120
./bpf_demo bpf_kernels.ptx

# Option 2: JIT — the driver API compiles PTX at load time
# (cuModuleLoadData handles this automatically)
nvcc -o bpf_demo bpf_ptx_host.cu -lcuda
./bpf_demo bpf_kernels.ptx
```

## PTX Crash Course (What You're Looking At)

### PTX vs SASS vs CUDA

```
CUDA C  →  nvcc  →  PTX (virtual ISA)  →  ptxas  →  SASS (real machine code)
                     ↑ YOU ARE HERE
```

- **PTX** is a virtual ISA — it looks like assembly but is hardware-independent
- **SASS** is the actual GPU machine code (different per SM generation)
- `ptxas` converts PTX → SASS. The Driver API can JIT this at load time
- Writing PTX lets you bypass `nvcc`'s optimizer and control instruction selection

### Register Types

```
.reg .u32  %r1;      // 32-bit unsigned int
.reg .s32  %r2;      // 32-bit signed int
.reg .u64  %rd1;     // 64-bit unsigned (pointers)
.reg .f32  %f1;      // 32-bit float (what we use everywhere)
.reg .f64  %fd1;     // 64-bit double
.reg .pred %p1;      // 1-bit predicate (for conditional execution)
```

### Key Instructions Used in BPF

| PTX Instruction | CUDA C Equivalent | Notes |
|----------------|-------------------|-------|
| `ld.global.f32` | `x = arr[i]` | Load from global memory |
| `st.global.f32` | `arr[i] = x` | Store to global memory |
| `ld.shared.f32` | `sdata[tid]` | Load from shared memory |
| `mad.lo.u32 %idx, %ctaid, %ntid, %tid` | `blockIdx.x * blockDim.x + threadIdx.x` | Thread index |
| `mul.wide.u32` | 32×32→64 multiply | Used for byte offset calc |
| `fma.rn.f32` | `a*b + c` (fused) | Single rounding, more precise |
| `ex2.approx.f32` | `exp2f(x)` / `__expf()` | Hardware exp2 unit |
| `lg2.approx.f32` | `log2f(x)` | Hardware log2 unit |
| `rcp.approx.f32` | `1.0f / x` | Hardware reciprocal (SFU) |
| `max.f32` | `fmaxf(a, b)` | Float max |
| `bar.sync 0` | `__syncthreads()` | Block barrier |
| `atom.global.add.f32` | `atomicAdd()` | Atomic float add |
| `atom.global.cas.b32` | `atomicCAS()` | Compare-and-swap |
| `setp.lt.f32 %p, a, b` | `if (a < b)` | Set predicate |
| `@%p instruction` | conditional exec | Predicated (branchless) |

### How exp() Works in PTX (Demystified)

In CUDA C you write `__expf(x)`. The compiler generates:

```ptx
// exp(x) = exp2(x * log2(e))
mul.f32         %f, %x, 0f3FB8AA3B;   // x * 1.4426950 (log2(e))
ex2.approx.f32  %f, %f;                // hardware exp2 in SFU
```

The GPU has a Special Function Unit (SFU) that computes `exp2`, `log2`, `rcp`,
`rsqrt`, and `sin`/`cos` in hardware. Everything else is built on top of these.

### How the Reduction Works (Step by Step)

The shared-memory tree reduction for `bpf_reduce_max`:

```
Thread:    0    1    2    3    4    5    6    7     (blockDim=8)
sdata:   [3.1, 0.5, 2.8, 1.0, 4.2, 0.1, 3.5, 2.0]

Step s=4: threads 0-3 compare with threads 4-7
sdata:   [4.2, 0.5, 3.5, 2.0, -, -, -, -]

Step s=2: threads 0-1 compare with threads 2-3
sdata:   [4.2, 2.0, -, -, -, -, -, -]

Step s=1: thread 0 compares with thread 1
sdata:   [4.2, -, -, -, -, -, -, -]

Thread 0 → atomicMax to global scalar
```

### How atomicMax for Float Works

CUDA doesn't have native `atomicMax` for float. We use CAS (compare-and-swap):

```ptx
// Reinterpret float bits as int for CAS
LOOP:
    mov.b32  %r_assumed, %r_old;           // save old int bits
    mov.b32  %f_old, %r_assumed;           // reinterpret as float
    max.f32  %f_new, %f_val, %f_old;       // float max
    mov.b32  %r_new, %f_new;               // back to int bits
    atom.global.cas.b32 %r_old, [addr], %r_assumed, %r_new;
    setp.ne  %p, %r_old, %r_assumed;       // did someone else write?
    @%p bra  LOOP;                          // retry if so
```

This works because IEEE 754 floats (positive) have the same ordering as integers.

### Predicated Execution (Branchless on GPU)

Instead of branching, PTX uses predicates:

```ptx
setp.lt.f32   %p_lt, %f_cdf_mid, %f_target;   // set predicate
@%p_lt  add.s32 %r_lo, %r_mid, 1;              // execute only if true
@!%p_lt mov.s32 %r_hi, %r_mid;                 // execute only if false
```

This avoids warp divergence — all threads execute both instructions but
only one actually writes. Critical for the binary search in resampling.

## What We Built (and What's Left)

### Implemented in PTX

1. **PCG32 PRNG** — Full PCG XSH-RR in PTX (64-bit state, 16 bytes/particle).
   Seeded per-particle with unique odd increments.

2. **Inverse CDF Normal Generation** — Full Acklam rational approximation
   (degree-6/5 central, degree-6/4 tails). ~1e-9 relative accuracy in float32.

3. **Fused Propagate+Weight Kernel** — OU transition, Student-t state noise
   (via host-pregenerated chi² from cuRAND), and Student-t observation
   log-weights with precomputed lgamma constant. One kernel launch per tick.

4. **All 13 BPF Kernels in PTX** — init_rng, init_particles,
   propagate_weight, set_scalar, reduce_max, reduce_sum, exp_sub, scale_wh,
   compute_loglik, resample, compute_var, gen_noise, silverman_jitter.

### Still Using CUDA C / Libraries

1. **Thrust Prefix Scan** — Resampling CDF still uses `thrust::inclusive_scan`.
   Writing a Blelloch scan in PTX is ~200 lines but would eliminate the last
   library dependency in the BPF path.

2. **cuRAND for Chi² Generation** — Bulk normal generation + square-sum for
   Student-t state noise. Fast (adds ~10μs/tick) but means the BPF isn't
   fully cuRAND-free when nu_state > 0.

3. **Warp-level Primitives** — SM_120 supports `shfl.sync` for warp-level
   reductions. The current reduce kernels use shared memory; warp shuffles
   would save ~30% on the reduction passes.

4. **APF / IMM** — The Auxiliary Particle Filter and Interacting Multiple
   Model filter remain nvcc-compiled. They share the same test harness but
   use cuRAND throughout.

## SM_120 (Blackwell) Specifics

- 128 CUDA cores per SM, up from 128 on Ada (SM_100)
- Native `atom.global.add.f32` — no CAS needed for float atomicAdd
- `ex2.approx.f32` precision: ~23 bits mantissa (same as SM_100)
- 256 KB shared memory per SM (configurable L1/shared split)
- Max 1024 threads per block, 32 warps per SM
- PTX ISA 8.8+ required for SM_120 features

## The Pipeline (CUDA C → PTX Mapping)

```
CUDA C                           PTX Kernel
─────────────────────────────    ─────────────────────────
bpf_propagate_weight<<<>>>  →    bpf_propagate_weight
bpf_set_scalar<<<1,1>>>     →    bpf_set_scalar
bpf_reduce_max<<<>>>        →    bpf_reduce_max
bpf_exp_sub_dev<<<>>>       →    bpf_exp_sub
bpf_set_scalar<<<1,1>>>     →    bpf_set_scalar
bpf_reduce_sum<<<>>>        →    bpf_reduce_sum
bpf_scale_wh_dev<<<>>>      →    bpf_scale_wh
bpf_set_scalar<<<1,1>>>     →    bpf_set_scalar
bpf_reduce_sum<<<>>>        →    bpf_reduce_sum
bpf_compute_loglik<<<1,1>>> →    bpf_compute_loglik
thrust::inclusive_scan       →    [still thrust — scan is hard]
bpf_resample<<<>>>          →    bpf_resample
```

## License

Open source — educational use. Do whatever you want with it.