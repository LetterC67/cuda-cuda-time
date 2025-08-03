#include <bits/stdc++.h>
using namespace std;

const uint THREAD_WORK  = 16;
const uint BLOCK_SIZE   = 512;
const uint WARP_SIZE    = 32;
const uint WARP_COUNT   = BLOCK_SIZE / WARP_SIZE - 1;            
constexpr uint WARP_WORK  = WARP_SIZE * THREAD_WORK;             
constexpr uint BLOCK_WORK = BLOCK_SIZE * THREAD_WORK - WARP_WORK;

constexpr int WARMUP_RUNS = 0;
constexpr int TIMED_RUNS  = 1;

#define ATOL 1e-5f
#define RTOL 1e-5f

#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        cerr << "CUDA Runtime Error at: " << file << ":" << line << endl;
        cerr << cudaGetErrorString(err) << " " << func << endl;
    }
}

template <typename T>
__device__ void device_swap(T& a, T& b) { T t = a; a = b; b = t; }

inline bool nearly_equal(float a, float b) {
    float diff = fabs(a - b);
    return diff <= ATOL + RTOL * fabs(b);
}

__global__ void __maxnreg__(32)
prefix_sum(const float* input, float* output, unsigned long long *prev, int *reg, unsigned N) {
    const int tid    = threadIdx.x;
    const int lane   = tid % 32;
    const uint lane8 = tid % 8;
    const uint warp_id = tid / 32;

    __shared__ float warp_sum[WARP_COUNT];
    __shared__ int bid;
    __shared__ float s_exclusive_sum;

    if (tid == 0) {
        bid = atomicAdd(reg, 1);
    }
    __syncthreads();

    const uint group8 = (lane / 8);
    input  += bid * BLOCK_WORK + WARP_WORK * warp_id + group8 * 8 * THREAD_WORK;
    output += bid * BLOCK_WORK + WARP_WORK * warp_id + group8 * 8 * THREAD_WORK;

    float vals[THREAD_WORK + 1], val;

    if (warp_id < WARP_COUNT) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint index = (lane8 ^ i) * 4 + 32 * i;
            reinterpret_cast<float4*>(&vals[i * 4])[0] = reinterpret_cast<const float4*>(&input[index])[0];
        }

        if (tid & 1) {
            device_swap(reinterpret_cast<float4*>(&vals[0])[0],  reinterpret_cast<float4*>(&vals[4])[0]);
            device_swap(reinterpret_cast<float4*>(&vals[8])[0],  reinterpret_cast<float4*>(&vals[12])[0]);
        }
        if (tid & 2) {
            device_swap(reinterpret_cast<float4*>(&vals[0])[0],  reinterpret_cast<float4*>(&vals[8])[0]);
            device_swap(reinterpret_cast<float4*>(&vals[4])[0],  reinterpret_cast<float4*>(&vals[12])[0]);
        }

        const uint base = group8 * 8 + ((lane & 1) << 2) + (lane8 >> 1);
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            vals[i] = __shfl_sync(-1, vals[i], base ^ (i / 4), 8);
        }
        __syncwarp();

        #pragma unroll
        for (int d = 2; d <= (int)THREAD_WORK; d <<= 1) {
            for (int i = d - 1; i < (int)THREAD_WORK; i += d) {
                vals[i] += vals[i - d / 2];
            }
        }

        {
            float vx = vals[THREAD_WORK - 1];
            for (uint mask = 1; mask < 32; mask <<= 1) {
                uint mask_full = (mask << 1) - 1;
                float t = __shfl_up_sync(-1, vx, mask);
                if ((lane & mask_full) == mask_full) {
                    vx += t;
                }
            }
            vals[THREAD_WORK - 1] = vx;
        }

        if (lane == 31) {
            warp_sum[warp_id] = vals[THREAD_WORK - 1];
        }
    }

    __syncthreads();

    if (warp_id == WARP_COUNT) {
        float vx = (lane < (int)WARP_COUNT) ? warp_sum[lane] : 0.0f;

        for (int d = 1; d < 32; d <<= 1) {
            float temp = __shfl_up_sync(-1, vx, d);
            if (lane >= d) vx += temp;
        }

        if (lane == (int)WARP_COUNT - 1) {
            prev[bid] = 0x0000000100000000ULL | (unsigned long long)__float_as_uint(vx);
            s_exclusive_sum = 0.0f;
        }
        if (lane < (int)WARP_COUNT) {
            warp_sum[lane] = vx;
        }

        if (bid > 0) {
            float exclusive_sum = 0.0f;
            for (int prev_block = bid - 1; prev_block >= 0; prev_block -= 32) {
                unsigned long long load = 0ULL;
                int load_block = prev_block - (31 - lane);
                while (load_block >= 0 &&
                      ((load = atomicAdd(&prev[load_block], 0ULL)) & 0x0000000300000000ULL) == 0ULL) {}

                unsigned mask = __ballot_sync(-1, (load & 0x0000000200000000ULL) > 0ULL);
                int highest = !mask ? -1 : 31 - __clz(mask);
                float value = (lane < highest) ? 0.0f : __uint_as_float((unsigned)load);

                for (int offset = 16; offset; offset >>= 1) {
                    value += __shfl_down_sync(-1, value, offset);
                }
                if (lane == 0) exclusive_sum += value;
                if (highest >= 0) break;
            }

            if (lane == 0) {
                s_exclusive_sum = exclusive_sum;
                prev[bid] = 0x0000000200000000ULL |
                            (unsigned long long)__float_as_uint(exclusive_sum + warp_sum[WARP_COUNT - 1]);
            }
        }
    } else {
        if (lane < 31) {
            val = vals[THREAD_WORK - 1];
            for (uint x = 8; x >= 1; x >>= 1) {
                float t = __shfl_up_sync(__activemask(), val, x);
                if (lane > (int)x && (x & lane) == 0 && ((x - 1) & lane) == x - 1) {
                    val += t;
                }
            }
            vals[THREAD_WORK - 1] = val;
        }

        {
            vals[THREAD_WORK] = __shfl_up_sync(-1, vals[THREAD_WORK - 1], 1);
            if (lane == 0) vals[THREAD_WORK] = 0.0f;

            vals[7]  += vals[16]; vals[3]  += vals[16]; vals[1]  += vals[16]; vals[0]  += vals[16];
            vals[11] += vals[7];  vals[5]  += vals[3];  vals[9]  += vals[7];  vals[13] += vals[11];
            vals[8]  += vals[7];  vals[2]  += vals[1];  vals[4]  += vals[3];  vals[6]  += vals[5];
            vals[12] += vals[11]; vals[10] += vals[9];  vals[14] += vals[13];

            vals[THREAD_WORK - 1] = __shfl_down_sync(-1, vals[THREAD_WORK], 1);
            if (lane == 31) vals[THREAD_WORK - 1] = val;
        }

        const uint base = group8 * 8 + (lane8 % 4) * 2 + ((lane8 & 4) >> 2);
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            vals[i] = __shfl_sync(-1, vals[i], base ^ ((i / 4) * 2), 8);
        }

        if (tid & 2) {
            device_swap(reinterpret_cast<float4*>(&vals[0])[0],  reinterpret_cast<float4*>(&vals[8])[0]);
            device_swap(reinterpret_cast<float4*>(&vals[4])[0],  reinterpret_cast<float4*>(&vals[12])[0]);
        }
        if (tid & 1) {
            device_swap(reinterpret_cast<float4*>(&vals[0])[0],  reinterpret_cast<float4*>(&vals[4])[0]);
            device_swap(reinterpret_cast<float4*>(&vals[8])[0],  reinterpret_cast<float4*>(&vals[12])[0]);
        }
    }

    __syncthreads();

    if (warp_id < WARP_COUNT) {
        const float prev_warp = (warp_id ? warp_sum[warp_id - 1] : 0.0f) + s_exclusive_sum;

        for (int i = 0; i < 4; i++) {
            const uint index = (lane8 ^ i) * 4 + 32 * i;
            float4 load = reinterpret_cast<float4*>(&vals[i * 4])[0];
            load.x += prev_warp; load.y += prev_warp; load.z += prev_warp; load.w += prev_warp;
            reinterpret_cast<float4*>(&output[index])[0] = load;
        }
    }
}

extern "C" void solution(const float* input, float* output, unsigned N) {
    const int num_blocks = (int)((N + BLOCK_WORK - 1) / BLOCK_WORK);

    int *reg = nullptr;
    unsigned long long *prev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&reg, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&prev, (unsigned long long)num_blocks * sizeof(unsigned long long)));

    dim3 grid(num_blocks), block(BLOCK_SIZE);

    auto run_once = [&]() {
        CUDA_CHECK(cudaMemset(reg, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(prev, 0, (unsigned long long)num_blocks * sizeof(unsigned long long)));
        prefix_sum<<<grid, block>>>(input, output, prev, reg, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    for (int i = 0; i < WARMUP_RUNS; ++i) run_once();

    float total_ms = 0.0f;
    for (int i = 0; i < TIMED_RUNS; ++i) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        run_once();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        printf("Run %d: %.3f ms\n", i + 1, ms);
    }
    
    printf("Avg time over %d timed runs: %.3f ms\n", TIMED_RUNS, total_ms / TIMED_RUNS);

    CUDA_CHECK(cudaFree(prev));
    CUDA_CHECK(cudaFree(reg));
}

static void cpu_prefix_sum(const vector<float>& in, vector<float>& out) {
    out.resize(in.size());
    double acc = 0.0;
    for (int i = 0; i < (int)in.size(); ++i) {
        acc += (double)in[i];
        out[i] = (float)acc;
    }
}

int main() {
    const int N = (1024 * 1024 * 1024 + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    vector<float> h_input(N);
    srand(12345);

    printf("Generating %d random floats...\n", N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }

    vector<float> h_output(N, 0.0f);
    vector<float> h_ref;

    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_input,  (unsigned long long)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, (unsigned long long)N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), (unsigned long long)N * sizeof(float), cudaMemcpyHostToDevice));

    solution(d_input, d_output, (unsigned)N);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, (unsigned long long)N * sizeof(float), cudaMemcpyDeviceToHost));

    cpu_prefix_sum(h_input, h_ref);
    int first_bad = N;
    for (int i = 0; i < N; ++i) {
        if (!nearly_equal(h_output[i], h_ref[i])) { first_bad = i; break; }
    }
    if (first_bad != N) {
        printf("Mismatch at %d: got %.8f, expected %.8f (atol=%.2e, rtol=%.2e)\n",
                     first_bad, h_output[first_bad], h_ref[first_bad], (double)ATOL, (double)RTOL);
        return 1;
    }
    printf("Reference check: OK (atol=%.2e, rtol=%.2e)\n", (double)ATOL, (double)RTOL);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}
