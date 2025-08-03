#include <thrust/merge.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << "\n";                         \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

int main() {
    const size_t array_size = 100'000'000;

    int* h_A      = static_cast<int*>(std::malloc(array_size * sizeof(int)));
    int* h_B      = static_cast<int*>(std::malloc(array_size * sizeof(int)));
    int* h_result = static_cast<int*>(std::malloc(2 * array_size * sizeof(int)));
    if (!h_A || !h_B || !h_result) {
        std::cerr << "Host allocation failed\n";
        return EXIT_FAILURE;
    }

    int indexA = 0, indexB = 0;
    for (uint i = 0; i < array_size + array_size; i++) {
        if ((indexB == array_size) || (rand() & 1 && indexA < array_size)) {
            h_A[indexA++] = i;
        } else if (indexB < array_size) {
            h_B[indexB++] = i;
        }
    }

    int *d_A, *d_B, *d_result;
    CUDA_CHECK(cudaMalloc(&d_A,      array_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_B,      array_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result, 2 * array_size * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, array_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, array_size * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    thrust::merge(
        thrust::device,
        d_A, d_A + array_size,
        d_B, d_B + array_size,
        d_result
    );
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Thrust merge of " << array_size << " + " << array_size
              << " elements took " << ms << " ms\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_result));
    std::free(h_A);
    std::free(h_B);
    std::free(h_result);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
