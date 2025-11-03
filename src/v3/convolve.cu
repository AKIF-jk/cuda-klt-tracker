/*********************************************************************
 * convolve.cu
 *********************************************************************/
/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>   /* malloc(), realloc() */
#include <stdio.h>

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"   /* printing */
#include "klt_cuda_mem.h"
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
__constant__ float d_gauss[MAX_KERNEL_WIDTH];
__constant__ float d_gaussDeriv[MAX_KERNEL_WIDTH];
static float sigma_last = -10.0;
static float total_time =0.0f;

/*********************************************************************
 * _KLTToFloatImage
 *
 * Given a pointer to image data (probably unsigned chars), copy
 * data to a float image.
 */

void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg)
{
  KLT_PixelType *ptrend = img + ncols*nrows;
  float *ptrout = floatimg->data;

  /* Output image must be large enough to hold result */
  assert(floatimg->ncols >= ncols);
  assert(floatimg->nrows >= nrows);

  floatimg->ncols = ncols;
  floatimg->nrows = nrows;

  while (img < ptrend)  *ptrout++ = (float) *img++;
}


/*********************************************************************
 * _computeKernels
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

  /* Compute kernels, and automatically determine widths */
  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float) (sigma*exp(-0.5f));
	
    /* Compute gauss and deriv */
    for (i = -hw ; i <= hw ; i++)  {
      gauss->data[i+hw]      = (float) exp(-i*i / (2*sigma*sigma));
      gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }

    /* Compute widths */
    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gauss->data[i+hw] / max_gauss) < factor ; 
         i++, gauss->width -= 2);
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor ; 
         i++, gaussderiv->width -= 2);
    if (gauss->width == MAX_KERNEL_WIDTH || 
        gaussderiv->width == MAX_KERNEL_WIDTH)
      KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
               "a sigma of %f", MAX_KERNEL_WIDTH, sigma);
  }

  /* Shift if width less than MAX_KERNEL_WIDTH */
  for (i = 0 ; i < gauss->width ; i++)
    gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
  for (i = 0 ; i < gaussderiv->width ; i++)
    gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];
  /* Normalize gauss and deriv */
  {
    const int hw = gaussderiv->width / 2;
    float den;
			
    den = 0.0;
    for (i = 0 ; i < gauss->width ; i++)  den += gauss->data[i];
    for (i = 0 ; i < gauss->width ; i++)  gauss->data[i] /= den;
    den = 0.0;
    for (i = -hw ; i <= hw ; i++)  den -= i*gaussderiv->data[i+hw];
    for (i = -hw ; i <= hw ; i++)  gaussderiv->data[i+hw] /= den;
  }

  sigma_last = sigma;
}
	

/*********************************************************************
 * _KLTGetKernelWidths
 *
 */

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width)
{
  _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}

template<int TX, int TY>
__global__ void convolve_horiz_tiled(
    const float* __restrict__ imgin,
    float*       __restrict__ imgout,
    int ncols,
    int nrows,
    int kwidth,
    int radius)
{
    extern __shared__ float smem[]; 

    const int x = blockIdx.x * TX + threadIdx.x; // column
    const int y = blockIdx.y * TY + threadIdx.y; // row

    if (y >= nrows) return;

    const int tileW = TX + 2 * radius;                  // shared row width
    const int smem_row = threadIdx.y * tileW;           // row offset in share memory
    const int lane = smem_row + threadIdx.x;            
    const int row_start = y * ncols;

    // Load center region (shifted by radius)
    if (x < ncols) {
        smem[lane + radius] = __ldg(&imgin[row_start + x]);
    } else {
        smem[lane + radius] = 0.0f;
    }

    // Load left halo 
    for (int hx = threadIdx.x; hx < radius; hx += blockDim.x) {
        int gx = blockIdx.x * TX + hx - radius;
        float v = 0.0f;
        if (gx >= 0 && gx < ncols) v = __ldg(&imgin[row_start + gx]);
        smem[smem_row + hx] = v;
    }

    // Load right halo
    for (int hx = threadIdx.x; hx < radius; hx += blockDim.x) {
        int gx = blockIdx.x * TX + TX + hx;
        float v = 0.0f;
        if (gx >= 0 && gx < ncols) v = __ldg(&imgin[row_start + gx]);
        smem[smem_row + radius + TX + hx] = v;
    }

    __syncthreads();

    if (x < ncols) {
        float sum = 0.0f;
        int base = smem_row + threadIdx.x; 
        for (int k = 0; k < kwidth; ++k) {
            float v  = smem[base + k];                // window element
            float kv = d_gauss[kwidth - 1 - k];       // reversed kernel element from constant memory
            sum += v * kv;
        }
        imgout[row_start + x] = sum;
    }
}


template<int TX, int TY>
__global__ void convolve_vert_tiled(
    const float* __restrict__ imgin,
    float*       __restrict__ imgout,
    int ncols,
    int nrows,
    int kwidth,
    int radius)
{
    extern __shared__ float smem[]; 

    const int col = blockIdx.x * TX + threadIdx.x; // column
    const int row = blockIdx.y * TY + threadIdx.y; // row

    if (col >= ncols) return;

    const int load_start = blockIdx.y * TY - radius;
    const int tileH = TY + 2 * radius;       // number of rows in shared tile
    const int smem_col_stride = TX;          // elements 
    for (int r = threadIdx.y; r < tileH; r += blockDim.y) {
        int global_r = load_start + r;
        int smem_idx = r * smem_col_stride + threadIdx.x;
        if (global_r < 0 || global_r >= nrows) {
            smem[smem_idx] = 0.0f;
        } else {
            smem[smem_idx] = __ldg(&imgin[global_r * ncols + col]); 
        }
    }

    __syncthreads();

    if (row < nrows) {
        int smem_center = (threadIdx.y + radius) * smem_col_stride + threadIdx.x;
        float sum = 0.0f;
        for (int k = 0; k < kwidth; ++k) {
            int offs = (k - radius) * smem_col_stride; 
            float v = smem[smem_center + offs];
            float kv = d_gaussDeriv[kwidth - 1 - k];   
            sum += v * kv;
        }
        imgout[row * ncols + col] = sum;
    }
}

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

  static float *d_in = NULL;
  static float *d_tmp = NULL; 
  static float *d_out = NULL; 
  static size_t cap_bytes = 0;

  if (bytes > cap_bytes) {
    if (d_in)  { cudaFree(d_in);  d_in  = NULL; }
    if (d_tmp) { cudaFree(d_tmp); d_tmp = NULL; }
    if (d_out) { cudaFree(d_out); d_out = NULL; }
    KLT_CUDA_ASSERT(cudaMalloc((void**)&d_in,  bytes));
    KLT_CUDA_ASSERT(cudaMalloc((void**)&d_tmp, bytes));
    KLT_CUDA_ASSERT(cudaMalloc((void**)&d_out, bytes));
    cap_bytes = bytes;
  }

  KLT_CUDA_ASSERT(cudaMemcpyAsync(d_in, imgin->data, bytes, cudaMemcpyHostToDevice));
  KLT_CUDA_ASSERT(cudaMemcpyToSymbol(d_gauss,      horiz_kernel.data, (size_t)horiz_kernel.width * sizeof(float)));
  KLT_CUDA_ASSERT(cudaMemcpyToSymbol(d_gaussDeriv, vert_kernel.data,  (size_t)vert_kernel.width * sizeof(float)));
  const int TX = 32;   
  const int TY = 8;    
  dim3 convBlock(TX, TY);
  dim3 convGrid((ncols + TX - 1) / TX, (nrows + TY - 1) / TY);

  const int hradius = horiz_kernel.width / 2;
  const int vradius = vert_kernel.width  / 2;

  size_t shmem_horiz = (size_t)TY * (size_t)(TX + 2 * hradius) * sizeof(float);
  size_t shmem_vert  = (size_t)(TY + 2 * vradius) * (size_t)TX * sizeof(float);

  convolve_horiz_tiled<TX, TY><<<convGrid, convBlock, shmem_horiz>>>(
      d_in, d_tmp, ncols, nrows, horiz_kernel.width, hradius);
  KLT_CUDA_ASSERT(cudaGetLastError());

  convolve_vert_tiled<TX, TY><<<convGrid, convBlock, shmem_vert>>>(
      d_tmp, d_out, ncols, nrows, vert_kernel.width, vradius);
  KLT_CUDA_ASSERT(cudaGetLastError());

  KLT_CUDA_ASSERT(cudaMemcpyAsync(imgout->data, d_out, bytes, cudaMemcpyDeviceToHost));

}

/*********************************************************************
 * _KLTComputeGradients
 */

void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady)
{
				
  /* Output images must be large enough to hold result */
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  /* Compute kernels, if necessary */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
	
  _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
  _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);

}
	

/*********************************************************************
 * _KLTComputeSmoothedImage
 */

void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth)
{
  /* Output image must be large enough to hold result */
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  /* Compute kernel, if necessary; gauss_deriv is not used */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}





