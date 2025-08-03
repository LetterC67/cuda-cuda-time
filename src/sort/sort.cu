#include <bits/stdc++.h>
using namespace std;

constexpr int WARMUP_RUNS = 0;
constexpr int TIMED_RUNS  = 1;

const     uint WARP_SIZE = 32;
const     uint RADIX_BITS = 8;
constexpr uint RADIX_SIZE = 1 << RADIX_BITS;

const     uint THREAD_WORK = 16;
const     uint CALCULATIVE_WARP_COUNT = 16;
const     uint LOOK_BACK_WARP_COUNT = 8;
constexpr uint BLOCK_WORK = CALCULATIVE_WARP_COUNT * 32 * THREAD_WORK;
constexpr uint BLOCK_SIZE = (CALCULATIVE_WARP_COUNT + LOOK_BACK_WARP_COUNT) * WARP_SIZE;

const     uint LOOK_BACK_FLAG_BITS = 3;
const     uint MAX_BROADCAST_DISTANCE = 3;
constexpr uint LOOK_BACK_SIZE = LOOK_BACK_WARP_COUNT * WARP_SIZE;
constexpr uint DATA_BITS = 32 - LOOK_BACK_FLAG_BITS;
constexpr uint COMPLETED_FLAG = 0xffffffffu << DATA_BITS;


const uint HISTOGRAM_BLOCK_SIZE = RADIX_SIZE;
const uint HISTOGRAM_BATCH = 8;

constexpr uint PREFIX_SUM_WORK = RADIX_SIZE / WARP_SIZE;

#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        cerr << "CUDA Runtime Error at: " << file << ":" << line << endl;
        cerr << cudaGetErrorString(err) << " " << func << endl;
    }
}

template <typename T>
__device__ void device_swap(T& a, T& b) { T temp = a; a = b; b = temp; }

__global__ void histogram(const uint *input, uint *histogram, uint N) {
    __shared__ uint shared_buckets[4][RADIX_SIZE];

    uint buffer[HISTOGRAM_BATCH];
    uint valid = 0;
    const uint tid = threadIdx.x;
    const uint bid = blockIdx.x;
    const uint HISTOGRAM_BLOCK_COUNT = gridDim.x;

    const uint H_BLOCK_WORK = ((N + HISTOGRAM_BLOCK_COUNT - 1) / HISTOGRAM_BLOCK_COUNT + 31) / 32 * 32;

    #pragma unroll
    for (uint i = 0; i < 4; i++) shared_buckets[i][tid] = 0;
    __syncthreads();

    const uint base  = bid * H_BLOCK_WORK;
    const uint limit = min(N, base + H_BLOCK_WORK);

    for (uint i = 0; i < H_BLOCK_WORK; i += HISTOGRAM_BLOCK_SIZE * HISTOGRAM_BATCH) {
        valid = 0;

        #pragma unroll
        for (uint j = 0; j < HISTOGRAM_BATCH; j++) {
            const uint index = i + j * HISTOGRAM_BLOCK_SIZE + tid + base;
            if (index < limit) {
                valid |= 1u << j;
                buffer[j] = input[index];
            } else {
                break;
            }
        }

        #pragma unroll
        for (uint j = 0; j < HISTOGRAM_BATCH; j++) {
            if (valid >> j & 1) {
                #pragma unroll
                for (uint offset = 0; offset < 32; offset += 8) {
                    atomicAdd(&shared_buckets[offset / 8][(buffer[j] >> offset) & 0xff], 1);
                }
            }
        }
    }

    __syncthreads();

    #pragma unroll
    for (uint offset = 0; offset < 32; offset += 8) {
        atomicAdd(&histogram[offset / 8 * RADIX_SIZE + tid], shared_buckets[offset / 8][tid]);
    }
}

__global__ void histogram_sum(uint *histogram) {
    histogram += RADIX_SIZE * blockIdx.x;
    const uint tid = threadIdx.x;
    __shared__ uint pref[RADIX_SIZE];
    pref[tid] = histogram[tid];
    __syncthreads();

    for (int d = 1; d < RADIX_SIZE; d <<= 1) {
        uint add;
        if (tid >= d) add = pref[tid - d];
        __syncthreads();
        if (tid >= d) pref[tid] += add;
        __syncthreads();
    }
    histogram[tid] = pref[tid];
}

template<uint OFFSET>
__device__ void block_prefsum(uint* vals, const uint *histogram, uint* warp_totals, uint *pref, uint *digit_offset, uint *block_histogram, uint bid) {
    const uint tid = threadIdx.x;
    const uint warp_id = tid / 32;
    const uint lane = tid % 32;
    uint temp;

    if (warp_id < CALCULATIVE_WARP_COUNT) {
        #pragma unroll
        for (int d = 2; d <= PREFIX_SUM_WORK; d <<= 1) {
            for (int i = d - 1; i < PREFIX_SUM_WORK; i += d) {
                vals[i] += vals[i - d / 2];
            }
        }

        uint val = vals[PREFIX_SUM_WORK - 1];
        for (uint mask = 1; mask < 2; mask <<= 1) {
            uint mask_full = (mask << 1) - 1;
            uint t = __shfl_up_sync(-1, val, mask);
            if ((lane & mask_full) == mask_full) val += t;
        }

        if (tid & 1) {
            block_histogram[tid / 2] = val;
            pref[bid * RADIX_SIZE + (tid / 2)] = (1u << DATA_BITS) | val;
        }

        for (uint mask = 2; mask < 32; mask <<= 1) {
            uint mask_full = (mask << 1) - 1;
            uint t = __shfl_up_sync(-1, val, mask);
            if ((lane & mask_full) == mask_full) val += t;
        }

        vals[PREFIX_SUM_WORK - 1] = val;
        if (lane == 31) warp_totals[warp_id] = vals[PREFIX_SUM_WORK - 1];
        digit_offset[0] = 0;
    }

    __syncthreads();

    uint prev_warp = 0;
    if (warp_id < CALCULATIVE_WARP_COUNT) {
        if (warp_id) {
            uint v = lane < 16 ? warp_totals[lane] : 0;
            for (int d = 1; d < 16; d <<= 1) {
                uint t = __shfl_up_sync(-1, v, d);
                if (lane >= d) v += t;
            }
            prev_warp = __shfl_sync(-1, v, warp_id - 1);
        }
    }

    uint val;
    if (warp_id < CALCULATIVE_WARP_COUNT) {
        if (lane < 31) {
            val = vals[PREFIX_SUM_WORK - 1];
            for (uint x = 8; x >= 1; x >>= 1) {
                uint t = __shfl_up_sync(__activemask(), val, x);
                if (lane > x && (x & lane) == 0 && ((x - 1) & lane) == x - 1) val += t;
            }
            vals[PREFIX_SUM_WORK - 1] = val;
        }
    }

    if (warp_id < CALCULATIVE_WARP_COUNT) {
        temp = __shfl_up_sync(-1, vals[PREFIX_SUM_WORK - 1], 1) + prev_warp;
        if (lane == 0) temp = prev_warp;

        vals[3] += temp;
        vals[1] += temp;
        vals[0] += temp;
        vals[7] += prev_warp;

        vals[5] += vals[3];
        vals[2] += vals[1];
        vals[4] += vals[3];
        vals[6] += vals[5];

        if (tid & 1) {
            if (tid != CALCULATIVE_WARP_COUNT * WARP_SIZE - 1)
                digit_offset[tid / 2 + 1] = -vals[7];
        }
    }
}

__device__ __forceinline__ uint f(uint pos, uint reg) {
    unsigned int m;
    unsigned int current_bit = 1 << pos;

    #if __CUDA_ARCH__ <= 860
    asm("{\n"
        "    .reg .pred p;\n"
        "    and.b32 %0, %1, %2;"
        "    setp.ne.u32 p, %0, 0;\n"
        "    vote.ballot.sync.b32 %0, p, 0xffffffff;\n"
        "    @!p not.b32 %0, %0;\n"
        "}\n"
        : "=r"(m)
        : "r"(reg), "r"(current_bit));
    #else
    asm (
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  .reg .u32  am;\n\t"
        "  and.b32 %0, %1, %2;\n\t"
        "  setp.ne.u32 p, %0, 0;\n\t"
        "  activemask.b32 am;\n\t"
        "  vote.sync.ballot.b32 %0, p, am;\n\t"
        "  @!p not.b32 %0, %0;\n\t"
        "}\n"
        : "=r"(m)
        : "r"(reg), "r"(current_bit)
    );
    #endif
    return m;
};

struct sort_shared_memory {
    uint warp_histogram[RADIX_SIZE * CALCULATIVE_WARP_COUNT + 16];
    uint tile_buf[BLOCK_WORK + 16];
    uint digit_offset[RADIX_SIZE];
    uint block_histogram[RADIX_SIZE];
};

template<uint OFFSET>
__global__ void sort(const uint *input, uint *output, uint N, uint* counter, uint *pref, uint *global_histogram) {
    const uint tid = threadIdx.x;
    const uint warp_id = tid / WARP_SIZE;
    const uint lane = threadIdx.x % 32;

    __shared__ uint _bid;
    if (tid == 0) _bid = atomicAdd(counter, 1);

    extern uint __shared__ _s[];
    sort_shared_memory *smem = reinterpret_cast<sort_shared_memory*>(_s);
    uint *warp_histogram  = smem->warp_histogram;
    uint *digit_offset    = smem->digit_offset;
    uint *tile_buf        = smem->tile_buf;
    uint *block_histogram = smem->block_histogram;

    uint vals[THREAD_WORK];
    uint warp_scan[PREFIX_SUM_WORK];

    for (int i = 0; i < CALCULATIVE_WARP_COUNT * RADIX_SIZE; i += BLOCK_SIZE * 4) {
        reinterpret_cast<uint4*>(&warp_histogram[i + tid * 4])[0] = make_uint4(0,0,0,0);
    }
    __syncthreads();

    uint bid = _bid;

    if (warp_id < CALCULATIVE_WARP_COUNT) {
        input += warp_id * (WARP_SIZE * THREAD_WORK) + bid * BLOCK_WORK;

        #pragma unroll
        for (uint i = 0; i < THREAD_WORK; i++) {
            const uint index = lane + i * WARP_SIZE;
            vals[i] = input[index];
        }
        #pragma unroll
        for (int j = 0; j < THREAD_WORK; j++) {
            atomicAdd(&warp_histogram[warp_id * RADIX_SIZE + ((vals[j] >> OFFSET) & 0xff) + (warp_id >= 8 ? 16 : 0)], 1);
        }
    }

    __syncthreads();

    if (warp_id < CALCULATIVE_WARP_COUNT) {
        for (uint i = 0; i < 8; i++) {
            const uint index = i + 8 * tid;
            warp_scan[i] = warp_histogram[index % CALCULATIVE_WARP_COUNT * RADIX_SIZE + index / CALCULATIVE_WARP_COUNT + ((tid & 1) ? 16 : 0)];
        }
    }

    block_prefsum<OFFSET>(warp_scan, warp_histogram, tile_buf, pref, digit_offset, block_histogram, bid);

    if (warp_id < CALCULATIVE_WARP_COUNT) {
        const uint idx = 8 * tid + 1;
        const uint base = idx / CALCULATIVE_WARP_COUNT + idx % CALCULATIVE_WARP_COUNT * RADIX_SIZE;

        #pragma unroll
        for (uint i = 0; i < 7; i++) {
            const uint index = i * RADIX_SIZE + base;
            warp_histogram[index] = warp_scan[i];
        }

        if (tid != CALCULATIVE_WARP_COUNT * WARP_SIZE - 1) {
            const uint index = 8 + 8 * tid;
            warp_histogram[index % CALCULATIVE_WARP_COUNT * RADIX_SIZE + (index / CALCULATIVE_WARP_COUNT)] = warp_scan[7];
        } else {
            warp_histogram[0] = 0;
        }
    }

    __syncthreads();

    if (warp_id >= CALCULATIVE_WARP_COUNT) {
        const uint lane2 = tid - CALCULATIVE_WARP_COUNT * WARP_SIZE;
        for (uint i = 0; i < RADIX_SIZE; i += LOOK_BACK_SIZE) {
            vals[i / LOOK_BACK_SIZE] = block_histogram[i + lane2];
        }
        if (bid != 0) {
            for (int i = 0; i < RADIX_SIZE; i += LOOK_BACK_SIZE) {
                const uint bit = i + lane2;
                const uint gload = bit ? global_histogram[bit - 1 + (OFFSET / 8) * RADIX_SIZE] : 0;
                uint ex = 0;

                int _prev_block = bid - 1;
                while (_prev_block >= 0) {
                    uint load = 0;
                    while(((load = atomicAdd(&pref[_prev_block * RADIX_SIZE + bit], 0)) & COMPLETED_FLAG) == 0) {}
                    uint value = load & (~COMPLETED_FLAG);
                    ex += value;

                    if ((load & COMPLETED_FLAG) == COMPLETED_FLAG) break;

                    _prev_block -= (load >> DATA_BITS);
                    if (bid - _prev_block <= MAX_BROADCAST_DISTANCE) {
                        pref[bid * RADIX_SIZE + bit] = ((bid - _prev_block) << DATA_BITS) | (ex + vals[i / LOOK_BACK_SIZE]);
                    }
                }
                pref[bid * RADIX_SIZE + bit] = COMPLETED_FLAG | (ex + vals[i / LOOK_BACK_SIZE]);
                digit_offset[bit] += ex + gload;
            }
        } else {
            for (int i = 0; i < RADIX_SIZE; i += LOOK_BACK_SIZE) {
                const uint bit = i + lane2;
                const uint gload = bit ? global_histogram[bit - 1 + (OFFSET / 8) * RADIX_SIZE] : 0;
                digit_offset[bit] += gload;
            }
        }
    } else {
        const uint base = RADIX_SIZE * warp_id;
        uint mask[8];

        #pragma unroll
        for (uint pos = OFFSET; pos < OFFSET + 8; pos++) mask[pos - OFFSET] = f(pos, vals[0]);

        mask[3] &= mask[2]; mask[7] &= mask[6]; mask[1] &= mask[0];
        mask[3] &= mask[1]; mask[5] &= mask[4]; mask[7] &= mask[3]; mask[7] &= mask[5];

        #pragma unroll
        for (int i = 0; i < THREAD_WORK; i++) {
            const uint _mask = mask[7];

            if (i != THREAD_WORK - 1) {
                for (uint pos = 0; pos < 4; pos++) mask[pos] = f(OFFSET + pos, vals[i + 1]);
            }

            const uint x = (vals[i] >> OFFSET) & 0xff;
            const int pref_warp_count_index = base + x;
            uint &data = warp_histogram[pref_warp_count_index];
            uint acc;
            const uint leader = 31 - __clz(_mask);

            if (i != THREAD_WORK - 1) {
                for (uint pos = 4; pos < 8; pos++) mask[pos] = f(OFFSET + pos, vals[i + 1]);
            }

            if (lane == leader) acc = atomicAdd(&data, __popc(_mask));
            acc = __shfl_sync(-1, acc, leader);

            const uint pos = acc + __popc(_mask & ((1 << lane) - 1));
            mask[3] &= mask[2]; mask[7] &= mask[6]; mask[1] &= mask[0];
            mask[3] &= mask[1]; mask[5] &= mask[4]; mask[7] &= mask[3]; mask[7] &= mask[5];

            tile_buf[pos] = vals[i];
        }
    }

    __syncthreads();

    for (int i = 0; i < BLOCK_WORK; i += BLOCK_SIZE) {
        if (i + tid >= BLOCK_WORK) break;
        const uint v = tile_buf[i + tid];
        const uint digit = (v >> OFFSET) & 0xff;
        const uint pos = digit_offset[digit] + i + tid;
        output[pos] = v;
    }
}

int get_histogram_block_count() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return (prop.maxThreadsPerMultiProcessor / HISTOGRAM_BLOCK_SIZE) * prop.multiProcessorCount * 2;
}

void solve(uint *input, uint *output, uint N) {
    const uint block_count = (N + BLOCK_WORK - 1) / BLOCK_WORK;
    uint *pref;
    uint *counter;
    uint *hist;

    CUDA_CHECK(cudaMalloc((void **)&pref,  block_count * RADIX_SIZE * sizeof(uint)));
    CUDA_CHECK(cudaMalloc((void **)&counter, sizeof(uint)));
    CUDA_CHECK(cudaMalloc((void **)&hist, 4 * RADIX_SIZE * sizeof(uint)));
    CUDA_CHECK(cudaMemset(hist, 0, 4 * RADIX_SIZE * sizeof(uint)));

    histogram<<<get_histogram_block_count(), HISTOGRAM_BLOCK_SIZE>>>(input, hist, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    histogram_sum<<<4, 256>>>(hist);
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr uint mem_size = sizeof(sort_shared_memory);

    CUDA_CHECK(cudaMemset(pref, 0, block_count * RADIX_SIZE * sizeof(uint)));
    CUDA_CHECK(cudaMemset(counter, 0, sizeof(uint)));
    sort<0><<<block_count, BLOCK_SIZE, mem_size>>>(input, output, N, counter, pref, hist);

    CUDA_CHECK(cudaMemset(counter, 0, sizeof(uint)));
    CUDA_CHECK(cudaMemset(pref, 0, block_count * RADIX_SIZE * sizeof(uint)));
    CUDA_CHECK(cudaDeviceSynchronize());
    sort<8><<<block_count, BLOCK_SIZE, mem_size>>>(output, input, N, counter, pref, hist);

    CUDA_CHECK(cudaMemset(counter, 0, sizeof(uint)));
    CUDA_CHECK(cudaMemset(pref, 0, block_count * RADIX_SIZE * sizeof(uint)));
    CUDA_CHECK(cudaDeviceSynchronize());
    sort<16><<<block_count, BLOCK_SIZE, mem_size>>>(input, output, N, counter, pref, hist);

    CUDA_CHECK(cudaMemset(counter, 0, sizeof(uint)));
    CUDA_CHECK(cudaMemset(pref, 0, block_count * RADIX_SIZE * sizeof(uint)));
    CUDA_CHECK(cudaDeviceSynchronize());
    sort<24><<<block_count, BLOCK_SIZE, mem_size>>>(output, input, N, counter, pref, hist);

    CUDA_CHECK(cudaFree(pref));
    CUDA_CHECK(cudaFree(counter));
    CUDA_CHECK(cudaFree(hist));
}

int main() {
    CUDA_CHECK(cudaFuncSetAttribute(sort<0>,  cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
    CUDA_CHECK(cudaFuncSetAttribute(sort<8>,  cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
    CUDA_CHECK(cudaFuncSetAttribute(sort<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
    CUDA_CHECK(cudaFuncSetAttribute(sort<24>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));

    const int N = BLOCK_WORK * ((1024 * 1024 * 256 + BLOCK_WORK - 1) / BLOCK_WORK);

    printf("Generating %d random numbers for sorting...\n", N);
    vector<uint> h_init(N);
    {
        mt19937 rng(5);
        for (int i = 0; i < N; i++) h_init[i] = rng();
    }
    
    uint *d_A = nullptr, *d_B = nullptr, *d_init = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_A,    N * sizeof(uint)));
    CUDA_CHECK(cudaMalloc((void **)&d_B,    N * sizeof(uint)));
    CUDA_CHECK(cudaMalloc((void **)&d_init, N * sizeof(uint)));

    CUDA_CHECK(cudaMemcpy(d_init, h_init.data(), N * sizeof(uint), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A,    h_init.data(), N * sizeof(uint), cudaMemcpyHostToDevice));

    printf("Warming up...\n");

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        CUDA_CHECK(cudaMemcpy(d_A, d_init, N * sizeof(uint), cudaMemcpyDeviceToDevice));
        solve(d_A, d_B, N);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0.0f;
    for (int i = 0; i < TIMED_RUNS; ++i) {
        CUDA_CHECK(cudaMemcpy(d_A, d_init, N * sizeof(uint), cudaMemcpyDeviceToDevice));
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, 0));
        solve(d_A, d_B, N);
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaDeviceSynchronize());

        printf("Run %d: %.3f ms\n", i + 1, ms);
    }
    printf("Avg GPU radix sort time over %d runs: %.3f ms\n", TIMED_RUNS, total_ms / TIMED_RUNS);

    vector<uint> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_A, N * sizeof(uint), cudaMemcpyDeviceToHost));

    printf("Run sort...\n");

    vector<uint> h_ref = h_init;
    sort(h_ref.begin(), h_ref.end());
    
    printf("Check vs sort...\n");

    int first_bad = -1;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != h_ref[i]) { first_bad = i; break; }
    }
    if (first_bad >= 0) {
        fprintf(stderr, "Mismatch at %d: got %u, expected %u\n", first_bad, h_out[first_bad], h_ref[first_bad]);
        int s = max(0, first_bad - 8), e = min(N, first_bad + 9);
        for (int j = s; j < e; ++j) fprintf(stderr, "  [%d] %u   ref=%u\n", j, h_out[j], h_ref[j]);
        return 1;
    }
    printf("Check vs sort: OK\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_init));
    return 0;
}
