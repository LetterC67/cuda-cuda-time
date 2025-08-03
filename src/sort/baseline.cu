#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <vector>
#include <random>
#include <iostream>
#include <cuda_runtime.h>

static constexpr size_t N = 1024 * 1024 * 64;

int main() {
    std::vector<uint32_t> h_vec(N);
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    for (size_t i = 0; i < N; ++i) {
        h_vec[i] = dist(rng);
    }

    thrust::device_vector<uint32_t> d_vec = h_vec;

    cudaMemcpy(thrust::raw_pointer_cast(d_vec.data()),
               h_vec.data(),
               N * sizeof(uint32_t),
               cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    thrust::sort(thrust::device, d_vec.begin(), d_vec.end());
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Thrust radix sort of " << N << " uint32 elements took "
              << ms << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
