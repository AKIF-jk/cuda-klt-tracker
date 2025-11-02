#include <cuda_runtime.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void* kltMallocUnified(size_t bytes) {
    void* ptr = NULL;
    cudaError_t err = cudaMallocManaged(&ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged(%zu) failed: %s\n",
                bytes, cudaGetErrorString(err));
        return NULL;
    }
    return ptr;
}

void kltFreeUnified(void* ptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFree() failed: %s\n",
                cudaGetErrorString(err));
    }
}

void* kltMalloc(size_t bytes){
    void *d_in;
    cudaError_t err = cudaMalloc((void**)&d_in, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc(%zu) failed: %s\n",
                bytes, cudaGetErrorString(err));
        return NULL;
    }
    return d_in;
}

void kltFree(void* ptr){
    cudaFree(ptr);
}

void kltMemcpyHostToDevice(void* dst, const void* src, size_t bytes){
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyHostToDevice(%p, %p, %zu) failed: %s\n",
                dst, src, bytes, cudaGetErrorString(err));
    }
}

void kltMemcpyDeviceToHost(void* dst, const void* src, size_t bytes){
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyDeviceToHost(%p, %p, %zu) failed: %s\n",
                dst, src, bytes, cudaGetErrorString(err));
    }
}

#ifdef __cplusplus
}  // extern "C"
#endif
