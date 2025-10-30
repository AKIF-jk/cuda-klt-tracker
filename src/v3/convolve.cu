
/*********************************************************************
 * convolve2.cu (optimized GPU separable convolution)
 *
 * Optimizations:
 *  - Launch configuration: 2D blocks (32x8) tuned for warp utilization
 *  - Occupancy: modest register use, shared memory sized by kernel radius
 *  - Communication: keep data on device across both passes, single D2H copy
 *  - Memory hierarchy: constant memory for kernel, shared-memory tiling with halos,
 *    read-only cache loads (__ldg) for global reads; coalesced accesses
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"
#include <cuda_runtime.h>

#define MAX_KERNEL_WIDTH 	71

#ifndef KLT_CUDA_ASSERT
#define KLT_CUDA_ASSERT(call)                                                  \
do {                                                                           \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error at %s:%d (%s): %s\n",                      \
                __FILE__, __LINE__, #call, cudaGetErrorString(err));          \
        fflush(stderr);                                                        \
        abort();                                                               \
    }                                                                          \
} while (0)
#endif

typedef struct  {
  int width;
  float data[MAX_KERNEL_WIDTH];
}  ConvolutionKernel;

/* Kernels */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0f;

/* Constant memory for active kernel (updated per pass) */
__constant__ float cKernel[MAX_KERNEL_WIDTH];

/* Persistent CUDA context to avoid per-call alloc/free and enable reuse */
typedef struct {
  float *d_in;
  float *d_tmp;
  float *d_out;
  size_t cap_bytes;
  cudaStream_t stream;
} ConvGPUCtx;

static ConvGPUCtx g_ctx = { nullptr, nullptr, nullptr, 0, nullptr };

static void ensure_capacity(size_t bytes)
{
  if (g_ctx.stream == nullptr) {
    KLT_CUDA_ASSERT(cudaStreamCreateWithFlags(&g_ctx.stream, cudaStreamNonBlocking));
  }
  if (bytes <= g_ctx.cap_bytes) return;
  if (g_ctx.d_in) { cudaFree(g_ctx.d_in); g_ctx.d_in = nullptr; }
  if (g_ctx.d_tmp) { cudaFree(g_ctx.d_tmp); g_ctx.d_tmp = nullptr; }
  if (g_ctx.d_out) { cudaFree(g_ctx.d_out); g_ctx.d_out = nullptr; }
  KLT_CUDA_ASSERT(cudaMalloc((void**)&g_ctx.d_in, bytes));
  KLT_CUDA_ASSERT(cudaMalloc((void**)&g_ctx.d_tmp, bytes));
  KLT_CUDA_ASSERT(cudaMalloc((void**)&g_ctx.d_out, bytes));
  g_ctx.cap_bytes = bytes;
}

void _KLTReleaseGPU()
{
  if (g_ctx.d_in) { cudaFree(g_ctx.d_in); g_ctx.d_in = nullptr; }
  if (g_ctx.d_tmp) { cudaFree(g_ctx.d_tmp); g_ctx.d_tmp = nullptr; }
  if (g_ctx.d_out) { cudaFree(g_ctx.d_out); g_ctx.d_out = nullptr; }
  if (g_ctx.stream) { cudaStreamDestroy(g_ctx.stream); g_ctx.stream = nullptr; }
  g_ctx.cap_bytes = 0;
}

/*********************************************************************
 * _KLTToFloatImage
 */
void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg)
{
  KLT_PixelType *ptrend = img + ncols*nrows;
  float *ptrout = floatimg->data;

  assert(floatimg->ncols >= ncols);
  assert(floatimg->nrows >= nrows);

  floatimg->ncols = ncols;
  floatimg->nrows = nrows;

  while (img < ptrend)  *ptrout++ = (float) *img++;
}

/*********************************************************************
 * _computeKernels (same as CPU reference)
 */
static void _computeKernels(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv)
{
  const float factor = 0.01f;   /* for truncating tail */
  int i;

  assert(MAX_KERNEL_WIDTH % 2 == 1);
  assert(sigma >= 0.0);

  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float) (sigma*expf(-0.5f));
    for (i = -hw ; i <= hw ; i++)  {
      gauss->data[i+hw]      = (float) expf(-i*i / (2*sigma*sigma));
      gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }
    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabsf(gauss->data[i+hw] / max_gauss) < factor ; i++, gauss->width -= 2);
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabsf(gaussderiv->data[i+hw] / max_gaussderiv) < factor ; i++, gaussderiv->width -= 2);
    if (gauss->width == MAX_KERNEL_WIDTH || gaussderiv->width == MAX_KERNEL_WIDTH)
      KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for a sigma of %f", MAX_KERNEL_WIDTH, sigma);
  }

  for (i = 0 ; i < gauss->width ; i++)
    gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
  for (i = 0 ; i < gaussderiv->width ; i++)
    gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];
  {
    const int hw = gaussderiv->width / 2;
    float den = 0.0f;
    for (i = 0 ; i < gauss->width ; i++)  den += gauss->data[i];
    for (i = 0 ; i < gauss->width ; i++)  gauss->data[i] /= den;
    den = 0.0f;
    for (i = -hw ; i <= hw ; i++)  den -= i*gaussderiv->data[i+hw];
    for (i = -hw ; i <= hw ; i++)  gaussderiv->data[i+hw] /= den;
  }

  sigma_last = sigma;
}

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width)
{
  _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}

/*********************************************************************
 * Device kernels: shared-memory tiled separable convolution
 *********************************************************************/

/* Horizontal: tile size (TX x TY), with halo of radius R on X */
template<int TX, int TY>
__global__ void convolve_horiz_tiled(
  const float* __restrict__ imgin,
  float* __restrict__ imgout,
  int ncols,
  int nrows,
  int kwidth,
  int radius)
{
  extern __shared__ float smem[]; // size = TY * (TX + 2*radius)
  const int x = blockIdx.x * TX + threadIdx.x;
  const int y = blockIdx.y * TY + threadIdx.y;

  if (y >= nrows) return;

  const int tileW = TX + 2*radius;
  const int lane = threadIdx.y * tileW + threadIdx.x; // for coalesced stores

  // Global start index for this tile row
  const int rowStart = y * ncols;

  // Load main region
  if (x < ncols) {
    smem[lane + radius] = __ldg(&imgin[rowStart + x]);
  }

  // Load left halo
  for (int hx = threadIdx.x; hx < radius; hx += TX) {
    int gx = blockIdx.x * TX + hx - radius;
    float v = 0.0f;
    if (gx >= 0 && gx < ncols) v = __ldg(&imgin[rowStart + gx]);
    smem[threadIdx.y * tileW + hx] = v;
  }
  // Load right halo
  for (int hx = threadIdx.x; hx < radius; hx += TX) {
    int gx = blockIdx.x * TX + TX + hx;
    float v = 0.0f;
    if (gx >= 0 && gx < ncols) v = __ldg(&imgin[rowStart + gx]);
    smem[threadIdx.y * tileW + radius + TX + hx] = v;
  }

  __syncthreads();

  if (x >= ncols) return;

  // Border policy: zero out borders consistent with CPU
  if (x < radius || x >= (ncols - radius)) {
    imgout[rowStart + x] = 0.0f;
    return;
  }

  float sum = 0.0f;
  const int base = threadIdx.y * tileW + threadIdx.x; // position including halo offset when adding k
  // Reverse-order multiply like reference
  for (int k = 0; k < kwidth; ++k) {
    float v = smem[base + k];
    float kv = cKernel[kwidth - 1 - k];
    sum += v * kv;
  }
  imgout[rowStart + x] = sum;
}

/* Vertical: tile size (TX x TY), with halo of radius R on Y */
template<int TX, int TY>
__global__ void convolve_vert_tiled(
  const float* __restrict__ imgin,
  float* __restrict__ imgout,
  int ncols,
  int nrows,
  int kwidth,
  int radius)
{
  extern __shared__ float smem[]; // size = (TY + 2*radius) * TX
  const int x = blockIdx.x * TX + threadIdx.x;
  const int y = blockIdx.y * TY + threadIdx.y;

  if (x >= ncols) return;

  const int tileH = TY + 2*radius;
  const int lane = (threadIdx.y + radius) * TX + threadIdx.x;

  // Load main region
  if (y < nrows) {
    smem[lane] = __ldg(&imgin[y * ncols + x]);
  }

  // Load top halo
  for (int hy = threadIdx.y; hy < radius; hy += TY) {
    int gy = blockIdx.y * TY + hy - radius;
    float v = 0.0f;
    if (gy >= 0 && gy < nrows) v = __ldg(&imgin[gy * ncols + x]);
    smem[hy * TX + threadIdx.x] = v;
  }
  // Load bottom halo
  for (int hy = threadIdx.y; hy < radius; hy += TY) {
    int gy = blockIdx.y * TY + TY + hy;
    float v = 0.0f;
    if (gy >= 0 && gy < nrows) v = __ldg(&imgin[gy * ncols + x]);
    smem[(radius + TY + hy) * TX + threadIdx.x] = v;
  }

  __syncthreads();

  if (y >= nrows) return;

  if (y < radius || y >= (nrows - radius)) {
    imgout[y * ncols + x] = 0.0f;
    return;
  }

  float sum = 0.0f;
  const int base = threadIdx.x; // advance by TX per step
  int smem_row = (threadIdx.y) * TX; // starting row aligned so that adding k walks down
  for (int k = 0; k < kwidth; ++k) {
    float v = smem[(smem_row + k * TX) + base];
    float kv = cKernel[kwidth - 1 - k];
    sum += v * kv;
  }
  imgout[y * ncols + x] = sum;
}

/*********************************************************************
 * Host helpers
 *********************************************************************/

static void launch_horizontal(
  const float* d_in,
  float* d_out,
  int ncols,
  int nrows,
  const ConvolutionKernel& kernel,
  cudaStream_t stream)
{
  const int kwidth = kernel.width;
  const int radius = kwidth / 2;
  KLT_CUDA_ASSERT(cudaMemcpyToSymbolAsync(cKernel, kernel.data, sizeof(float) * kwidth, 0, cudaMemcpyHostToDevice, stream));

  const int TX = 32;
  const int TY = 8;
  dim3 block(TX, TY);
  dim3 grid((ncols + TX - 1) / TX, (nrows + TY - 1) / TY);
  size_t smem_bytes = (size_t)TY * (size_t)(TX + 2*radius) * sizeof(float);
  convolve_horiz_tiled<TX, TY><<<grid, block, smem_bytes, stream>>>(d_in, d_out, ncols, nrows, kwidth, radius);
  KLT_CUDA_ASSERT(cudaGetLastError());
}

static void launch_vertical(
  const float* d_in,
  float* d_out,
  int ncols,
  int nrows,
  const ConvolutionKernel& kernel,
  cudaStream_t stream)
{
  const int kwidth = kernel.width;
  const int radius = kwidth / 2;
  KLT_CUDA_ASSERT(cudaMemcpyToSymbolAsync(cKernel, kernel.data, sizeof(float) * kwidth, 0, cudaMemcpyHostToDevice, stream));

  const int TX = 32;
  const int TY = 8;
  dim3 block(TX, TY);
  dim3 grid((ncols + TX - 1) / TX, (nrows + TY - 1) / TY);
  size_t smem_bytes = (size_t)(TY + 2*radius) * (size_t)TX * sizeof(float);
  convolve_vert_tiled<TX, TY><<<grid, block, smem_bytes, stream>>>(d_in, d_out, ncols, nrows, kwidth, radius);
  KLT_CUDA_ASSERT(cudaGetLastError());
}

/*********************************************************************
 * Optimized _convolveSeparate using on-device intermediate
 *********************************************************************/
static void _convolveSeparate(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
  const int ncols = imgin->ncols;
  const int nrows = imgin->nrows;
  const size_t npixels = (size_t)ncols * (size_t)nrows;
  const size_t bytes = npixels * sizeof(float);

  assert(imgout->ncols >= ncols);
  assert(imgout->nrows >= nrows);

  ensure_capacity(bytes);

  // H2D once
  KLT_CUDA_ASSERT(cudaMemcpyAsync(g_ctx.d_in, imgin->data, bytes, cudaMemcpyHostToDevice, g_ctx.stream));

  // Horizontal pass: d_in -> d_tmp
  launch_horizontal(g_ctx.d_in, g_ctx.d_tmp, ncols, nrows, horiz_kernel, g_ctx.stream);

  // Vertical pass: d_tmp -> d_out
  launch_vertical(g_ctx.d_tmp, g_ctx.d_out, ncols, nrows, vert_kernel, g_ctx.stream);

  // D2H once
  KLT_CUDA_ASSERT(cudaMemcpyAsync(imgout->data, g_ctx.d_out, bytes, cudaMemcpyDeviceToHost, g_ctx.stream));
  KLT_CUDA_ASSERT(cudaStreamSynchronize(g_ctx.stream));
}

/*********************************************************************
 * Public APIs mirroring CPU
 *********************************************************************/
void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady)
{
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  if (fabsf(sigma - sigma_last) > 0.05f)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  const int ncols = img->ncols;
  const int nrows = img->nrows;
  const size_t bytes = (size_t)ncols * (size_t)nrows * sizeof(float);

  ensure_capacity(bytes);

  // Upload image once
  KLT_CUDA_ASSERT(cudaMemcpyAsync(g_ctx.d_in, img->data, bytes, cudaMemcpyHostToDevice, g_ctx.stream));

  // gradx: (gaussderiv then gauss)
  launch_horizontal(g_ctx.d_in,  g_ctx.d_tmp, ncols, nrows, gaussderiv_kernel, g_ctx.stream);
  launch_vertical  (g_ctx.d_tmp, g_ctx.d_out, ncols, nrows, gauss_kernel,      g_ctx.stream);
  KLT_CUDA_ASSERT(cudaMemcpyAsync(gradx->data, g_ctx.d_out, bytes, cudaMemcpyDeviceToHost, g_ctx.stream));

  // grady: (gauss then gaussderiv), reusing the same uploaded d_in
  launch_horizontal(g_ctx.d_in,  g_ctx.d_tmp, ncols, nrows, gauss_kernel,      g_ctx.stream);
  launch_vertical  (g_ctx.d_tmp, g_ctx.d_out, ncols, nrows, gaussderiv_kernel, g_ctx.stream);
  KLT_CUDA_ASSERT(cudaMemcpyAsync(grady->data, g_ctx.d_out, bytes, cudaMemcpyDeviceToHost, g_ctx.stream));

  KLT_CUDA_ASSERT(cudaStreamSynchronize(g_ctx.stream));
}

void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth)
{
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  if (fabsf(sigma - sigma_last) > 0.05f)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}





