#include <cstdio>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#define N (1024 * 1024 * 1024)

#define cudaCheck(err) \
    do { \
        cudaError_t e = (err); \
        if (e != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1); \
        } \
    } while (0)


int main() {
    float *d_in = nullptr, *d_out = nullptr;
    size_t   size = N * sizeof(float);
    cudaCheck(cudaMalloc(&d_in,  size));
    cudaCheck(cudaMalloc(&d_out, size));

    {
        thrust::device_ptr<float> dev_ptr(d_in);
        thrust::fill(dev_ptr, dev_ptr + N, 1.0f);
    }

    void    *d_temp_storage = nullptr;
    size_t   temp_bytes      = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_bytes, d_in, d_out, N);
    cudaCheck(cudaMalloc(&d_temp_storage, temp_bytes));

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));


    cudaCheck(cudaEventRecord(start, 0));
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_bytes, d_in, d_out, N);
    cudaCheck(cudaEventRecord(stop, 0));
    cudaCheck(cudaEventSynchronize(stop));

    float milliseconds = 0;
    cudaCheck(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("CUB exclusive scan of %d elements took %.3f ms\n", N, milliseconds);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_temp_storage);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

