#include <bits/stdc++.h>
using namespace std;

const     uint NUM_THREADS = 512;
const     uint THREAD_WORK = 15;
constexpr uint BLOCK_WORK  = NUM_THREADS * THREAD_WORK;

const int  WARMUP_RUNS  = 0;
const int  TIMED_RUNS   = 1;

#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        cerr << "CUDA Runtime Error at: " << file << ":" << line << endl;
        cerr << cudaGetErrorString(err) << " " << func << endl;
    }
}

__device__ uint co_rank(const int *a, const int *b, uint n, uint m, uint k) {
    if (!k) return 0;

    uint r = min(n, k);
    uint l = min(k < m ? 0u : k - m, r);

    uint lg = 31 - __clz(r);
    uint res = 0;
    for (int pos = lg; pos >= 0; pos--) {
        uint imm = res | (1u << pos);
        if (imm <= l || (imm <= r && a[imm - 1] <= b[k - imm])) {
            res |= (1u << pos);
        }
    }
    return res;
}

__device__ uint co_rank_shared(const int *a, const int *b, uint n, uint m, uint k) {
    if (!k) return 0;

    uint r = min(n, k);
    uint l = min(k < m ? 0u : k - m, r);

    while (l < r) {
        uint mid = (l + r + 1) >> 1;
        if (a[mid - 1] > b[k - mid]) {
            r = mid - 1;
        } else {
            l = mid;
        }
    }
    return l;
}

__device__ void merge_chunk(const int *sa, const int *sb, int *out, uint &ia, uint &ib) {
    #pragma unroll
    for (uint idx = 0; idx < THREAD_WORK; idx++) {
        if (sa[ia] <= sb[ib]) {
            out[idx] = sa[ia++];
        } else {
            out[idx] = sb[ib++];
        }
    }
}

__global__ void merge_path(const int *a, const int *b, uint n, uint m, uint *bounds) {
    const uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx * BLOCK_WORK < n + m) {
        bounds[tidx] = co_rank(a, b, n, m, (tidx + 1) * BLOCK_WORK);
    }
}

__global__ void merge(const int *a, const int *b, int *c, uint n, uint m, uint *bounds) {
    const uint bid = blockIdx.x;
    const uint tid = threadIdx.x;

    uint from_a = 0, to_a, from_b = 0, to_b;

    __shared__ int shared_mem[BLOCK_WORK + 1 + THREAD_WORK];
    int *sa = shared_mem;
    int *sb;

    if (bid > 0) {
        from_a = bounds[bid - 1];
    }
    to_a = bounds[bid];

    from_b = bid * BLOCK_WORK - from_a;
    to_b   = (bid + 1) * BLOCK_WORK - to_a;

    const uint len_a = to_a - from_a;
    const uint len_b = to_b - from_b;

    sb = shared_mem + len_a + THREAD_WORK;

    a += from_a;
    b += from_b;
    c += bid * BLOCK_WORK;

    if (tid < THREAD_WORK) {
        sa[len_a + tid] = INT_MAX;
        sb[len_b] = INT_MAX;
    }

    #pragma unroll
    for (int i = 0; i < THREAD_WORK; i++) {
        uint save_idx = i * NUM_THREADS + tid;
        bool from_b_src = (save_idx >= len_a);
        shared_mem[save_idx + (from_b_src ? THREAD_WORK : 0)] =
            !from_b_src ? a[save_idx] : b[save_idx - len_a];
    }

    __syncthreads();

    uint ia = co_rank_shared(sa, sb, len_a, len_b, tid * THREAD_WORK);
    uint ib = tid * THREAD_WORK - ia;

    int out[THREAD_WORK];
    merge_chunk(sa, sb, out, ia, ib);

    __syncthreads();

    #pragma unroll
    for (uint i = 0; i < THREAD_WORK; i++) {
        shared_mem[i + tid * THREAD_WORK] = out[i];
    }

    __syncthreads();

    for (uint i = 0; i < BLOCK_WORK; i += NUM_THREADS) {
        c[i + tid] = shared_mem[i + tid];
    }
}

int main() {
    const uint n = (100'000'000 + BLOCK_WORK - 1) / BLOCK_WORK * BLOCK_WORK;
    const uint m = (100'000'000 + BLOCK_WORK - 1) / BLOCK_WORK * BLOCK_WORK;
    const uint total = n + m;

    vector<int> h_a(n), h_b(m), h_c(total);

    uint ia = 0, ib = 0;
    for (uint i = 0; i < total; i++) {
        if ((ib == m) || ((rand() & 1) && ia < n)) {
            h_a[ia++] = static_cast<int>(i);
        } else if (ib < m) {
            h_b[ib++] = static_cast<int>(i);
        }
    }

    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    uint *d_bounds = nullptr;

    const uint num_tiles = (total + BLOCK_WORK - 1) / BLOCK_WORK;

    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b, m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c, total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bounds, num_tiles * sizeof(uint)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), m * sizeof(int), cudaMemcpyHostToDevice));

    dim3 mp_grid((num_tiles + 127) / 128);
    dim3 mp_block(128);
    dim3 merge_grid(num_tiles);
    dim3 merge_block(NUM_THREADS);

    auto run_once = [&]() {
        merge_path<<<mp_grid, mp_block>>>(d_a, d_b, n, m, d_bounds);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        merge<<<merge_grid, merge_block>>>(d_a, d_b, d_c, n, m, d_bounds);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    for (int i = 0; i < WARMUP_RUNS; i++) {
        run_once();
    }

    float total_ms = 0.;
    for (int i = 0; i < TIMED_RUNS; i++) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        run_once();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        printf("Run %d: %.3f ms\n", i + 1, ms);
    }

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, total * sizeof(int), cudaMemcpyDeviceToHost));

    {
        vector<int> h_ref(total);
        merge(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_ref.begin());
        bool same = equal(h_c.begin(), h_c.end(), h_ref.begin());
        if (!same) {
            printf("[ERROR] GPU result differs from merge reference!\n");
            for (uint i = 0; i < total; i++) {
                if (h_c[i] != h_ref[i]) {
                    printf("Mismatch at %u: got %d, expected %d\n",
                                 i, h_c[i], h_ref[i]);
                    break;
                }
            }
            return 1;
        } else {
            printf("Check vs merge: OK\n");
        }
    }

    printf("Avg time over %d timed runs: %.3f ms\n", TIMED_RUNS, total_ms / TIMED_RUNS);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_bounds));
    return 0;
}
