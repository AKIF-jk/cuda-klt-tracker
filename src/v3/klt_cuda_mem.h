#ifndef KLT_CUDA_MEMORY_H
#define KLT_CUDA_MEMORY_H

#include <stddef.h>   // for size_t

// ----- Unified Memory -----
#ifdef __cplusplus
extern "C" {
#endif

void* kltMallocUnified(size_t bytes);
void  kltFreeUnified(void* ptr);

#ifdef __cplusplus
}
#endif

// ----- Standard CUDA Memory -----
#ifdef __cplusplus
extern "C" {
#endif

void* kltMalloc(size_t bytes);
void  kltFree(void* ptr);
void  kltMemcpyHostToDevice(void* dst, const void* src, size_t bytes);
void  kltMemcpyDeviceToHost(void* dst, const void* src, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif // KLT_CUDA_MEMORY_H
