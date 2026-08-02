# Chapter 14: 自定义高性能Kernel编写模式 深度分析

## 1. 设计动机

**WHY自定义Kernel**: 库(cuBLAS/cuDNN)覆盖标准算子，但训练中有大量非标准计算:
RoPE变体、MoE routing、Custom loss、Flash Decoding等。
这些无法直接使用库，需要手写CUDA kernel达到peak性能。

**目标**: 掌握从问题分析到高性能kernel实现的完整方法论。

## 2. Kernel设计方法论

### 2.1 分析框架

```
Step 1: 确定Bound类型
  计算密度 = FLOPs / Bytes_accessed
  与ridge point (295 for H100 FP16) 对比
  
Step 2: 确定并行维度
  哪些维度可以独立并行? (batch, token, head, ...)
  哪些维度需要协作? (reduce, softmax的seq维)

Step 3: 选择tile策略
  Memory-bound: 最大化带宽利用(large vectorized loads)
  Compute-bound: 最大化计算复用(large tiles with SMEM)
  
Step 4: 设计数据流
  GMEM → (SMEM) → RF → Compute → RF → (SMEM) → GMEM
  每一步确定layout和访问pattern
```

### 2.2 Kernel骨架模板

```cpp
// Memory-bound kernel (elementwise/reduction):
template<typename T, int BLOCK, int VEC>
__global__ void memory_bound_kernel(T* out, const T* in, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    // Vectorized load/store
    using VecT = aligned_vector<T, VEC>;
    for (int i = tid; i < N / VEC; i += stride) {
        VecT val = reinterpret_cast<const VecT*>(in)[i];
        // Process val...
        reinterpret_cast<VecT*>(out)[i] = val;
    }
}

// Compute-bound kernel (GEMM-like):
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void compute_bound_kernel(...) {
    extern __shared__ char smem[];
    auto A_smem = reinterpret_cast<half*>(smem);
    auto B_smem = A_smem + TILE_M * TILE_K;
    
    float acc[TILE_M/WARP_M][TILE_N/WARP_N] = {0};
    
    for (int k = 0; k < K; k += TILE_K) {
        // Stage 1: Load GMEM → SMEM (async)
        load_tile_async(A_smem, A_gmem + k);
        load_tile_async(B_smem, B_gmem + k);
        cp_async_wait_all();
        __syncthreads();
        
        // Stage 2: Compute from SMEM
        for (int kk = 0; kk < TILE_K; kk += MMA_K) {
            mma_compute(acc, A_smem, B_smem, kk);
        }
        __syncthreads();
    }
    // Stage 3: Write results
    store_output(C_gmem, acc);
}
```

## 3. Pattern 1: Fused Elementwise

### 3.1 适用场景

```
多个逐元素操作串联:
LayerNorm → Dropout → GELU → Scale

每个单独都是memory-bound (计算密度~1-3)
融合后: 只需1次GMEM读 + 1次GMEM写
```

### 3.2 实现模式

```cpp
template<int VEC_SIZE=8>  // 8×half = 128 bits
__global__ void fused_ln_dropout_gelu(
    half* __restrict__ out,
    const half* __restrict__ in,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    float dropout_prob, uint64_t seed,
    int hidden_size) {
    
    // 每block处理一行 (一个token)
    int row = blockIdx.x;
    const half* row_in = in + row * hidden_size;
    half* row_out = out + row * hidden_size;
    
    // Phase 1: 计算mean和var (online welford)
    float mean = 0.f, var = 0.f;
    int count = 0;
    for (int i = threadIdx.x * VEC_SIZE; i < hidden_size; 
         i += blockDim.x * VEC_SIZE) {
        half8 val = load_vec<8>(row_in + i);
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            float x = __half2float(val[v]);
            count++;
            float delta = x - mean;
            mean += delta / count;
            var += delta * (x - mean);
        }
    }
    // Warp reduce + block reduce
    block_reduce_mean_var(&mean, &var, count);
    float inv_std = rsqrtf(var / hidden_size + 1e-5f);
    
    // Phase 2: Apply LN + Dropout + GELU (fused write)
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, row * hidden_size + threadIdx.x, 0, &rng);
    
    for (int i = threadIdx.x * VEC_SIZE; i < hidden_size;
         i += blockDim.x * VEC_SIZE) {
        half8 val = load_vec<8>(row_in + i);
        half8 result;
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            float x = __half2float(val[v]);
            // LayerNorm
            x = (x - mean) * inv_std;
            x = x * __half2float(gamma[i+v]) + __half2float(beta[i+v]);
            // Dropout
            float mask = (curand_uniform(&rng) > dropout_prob) ? 1.f : 0.f;
            x *= mask / (1.f - dropout_prob);
            // GELU
            x = x * 0.5f * (1.f + erff(x * 0.7071067811865476f));
            result[v] = __float2half(x);
        }
        store_vec<8>(row_out + i, result);
    }
}
```

### 3.3 性能分析

```
hidden_size = 4096, seq_len × batch = 8192 tokens:
数据量: 8192 × 4096 × 2B = 64 MB (读) + 64 MB (写) = 128 MB
理论时间: 128 MB / 3.35 TB/s = 38 μs
实测时间: ~45 μs (85% bandwidth efficiency)

对比unfused (4个kernel):
理论时间: 128 MB × 4 / 3.35 TB/s = 153 μs
加速比: 3.4×
```

## 4. Pattern 2: Online Reduction

### 4.1 Softmax实现

```cpp
// 单pass online softmax (FlashAttention风格)
template<int BLOCK=256, int VEC=8>
__global__ void online_softmax(
    half* out, const half* in, int N) {
    
    int row = blockIdx.x;
    const half* row_in = in + row * N;
    
    // 每thread维护局部 (max, sum)
    float thread_max = -INFINITY;
    float thread_sum = 0.f;
    
    // Pass 1: online max + sum
    for (int i = threadIdx.x * VEC; i < N; i += BLOCK * VEC) {
        half8 val = load_vec<8>(row_in + i);
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            float x = __half2float(val[v]);
            float old_max = thread_max;
            thread_max = fmaxf(thread_max, x);
            // Rescale previous sum
            thread_sum = thread_sum * expf(old_max - thread_max) 
                       + expf(x - thread_max);
        }
    }
    
    // Block reduce (max, sum) with rescaling
    __shared__ float s_max, s_sum;
    block_reduce_online_softmax(&s_max, &s_sum, 
                                 thread_max, thread_sum);
    
    // Pass 2: normalize and write
    float global_max = s_max;
    float global_sum = s_sum;
    
    for (int i = threadIdx.x * VEC; i < N; i += BLOCK * VEC) {
        half8 val = load_vec<8>(row_in + i);
        half8 result;
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            float x = __half2float(val[v]);
            result[v] = __float2half(
                expf(x - global_max) / global_sum);
        }
        store_vec<8>(out + row * N + i, result);
    }
}
```

## 5. Pattern 3: Warp-Level Primitives

### 5.1 Warp Shuffle Reduction

```cpp
// 比SMEM reduction更高效:
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;  // 所有lane都得到sum
}

__device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block reduce using warp primitives:
__device__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[32];  // max 32 warps
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    val = warp_reduce_sum(val);
    if (lane_id == 0) warp_sums[warp_id] = val;
    __syncthreads();
    
    // Final reduce in first warp
    val = (threadIdx.x < blockDim.x / 32) ? warp_sums[lane_id] : 0.f;
    if (warp_id == 0) val = warp_reduce_sum(val);
    return val;  // 只有thread 0有最终结果
}
```

## 6. Pattern 4: Multi-Stage Pipeline (SM90)

```cpp
// 带TMA的multi-stage GEMM pipeline:
template<int STAGES=5>
__global__ void pipelined_gemm(...) {
    extern __shared__ char smem[];
    
    // Circular buffer in SMEM
    half* A_smem[STAGES];
    half* B_smem[STAGES];
    for (int s = 0; s < STAGES; s++) {
        A_smem[s] = (half*)(smem + s * stage_bytes);
        B_smem[s] = A_smem[s] + tile_A_size;
    }
    
    // 用mbarrier做同步
    __shared__ uint64_t mbar[STAGES];
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; s++)
            mbarrier_init(&mbar[s], 1);
    }
    __syncthreads();
    
    // Prologue: 预填充前STAGES个stage
    for (int s = 0; s < STAGES; s++) {
        tma_load_async(A_smem[s], A_gmem + s * TILE_K, &mbar[s]);
        tma_load_async(B_smem[s], B_gmem + s * TILE_K, &mbar[s]);
    }
    
    // Main loop
    float acc[...] = {0};
    for (int k = 0; k < K_TILES; k++) {
        int stage = k % STAGES;
        int next_stage = (k + STAGES) % STAGES;
        
        // Wait for current stage data
        mbarrier_wait(&mbar[stage]);
        
        // Compute on current stage
        wgmma_compute(acc, A_smem[stage], B_smem[stage]);
        
        // Prefetch next tile into reused stage
        if (k + STAGES < K_TILES) {
            tma_load_async(A_smem[next_stage], 
                          A_gmem + (k+STAGES) * TILE_K, 
                          &mbar[next_stage]);
        }
    }
    
    // Epilogue
    wgmma_wait_all();
    store_output(C_gmem, acc);
}
```

## 7. Pattern 5: Split-K 与 Stream-K

### 7.1 Split-K

```
标准GEMM: 每CTA处理完整的K维度
问题: M×N很小时, block太少, SM利用不足

Split-K: K维度切分, 多个CTA并行做部分K
最终原子加/单独reduce

Grid: (M_tiles × N_tiles × split_k)
每CTA只计算 K/split_k 的部分累加
最后atomicAdd到output

WHY需要: 小batch inference, M和N小但K大
```

### 7.2 Stream-K (CUTLASS)

```
Stream-K: 更灵活的工作划分
- 将所有work (M_tiles × N_tiles × K_tiles) 均匀分到SM
- 每SM可以处理不完整的tile
- 消除split-K的额外atomic + reduce开销

优势:
- 完美负载均衡 (无wave quantization)
- 无额外reduce kernel
- 自适应各种shape

代价:
- 实现复杂 (partial tile处理)
- 需要原子操作处理tile边界
```

## 8. 性能优化Checklist

```
□ Memory-bound kernel:
  □ 128-bit vectorized loads (float4 / half8)
  □ Coalesced access pattern
  □ 每thread处理多元素(增加ILP)
  □ Grid-stride loop (避免wave quantization)
  □ 避免warp divergence

□ Compute-bound kernel:
  □ 使用Tensor Core (WGMMA/HMMA)
  □ 多stage pipeline (3-7 stages)
  □ SMEM使用Swizzle (无bank conflict)
  □ Register pressure < 160
  □ Persistent grid (L2 reuse)

□ 通用:
  □ 编译使用 -use_fast_math (当精度允许)
  □ 使用 __restrict__ 提示编译器
  □ #pragma unroll 关键循环
  □ 避免条件分支 (predicated execution替代)
  □ Profile验证 (ncu --set full)
```

## 9. 总结

```
高性能Kernel编写核心能力:
1. 识别Bound类型 → 选择正确策略
2. 设计数据流 → GMEM→SMEM→RF→Compute
3. 选择并行粒度 → Block/Warp/Thread mapping
4. 实现Pipeline → 隐藏所有延迟
5. Profile验证 → 量化优化效果

从零到高性能的路径:
Naive → Vectorize → Coalesce → Tile(SMEM) → Pipeline → TC利用
每步带来数量级或数十%提升
```
