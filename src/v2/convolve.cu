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
#include <cuda_runtime.h>

#define MAX_KERNEL_WIDTH 	71

/* Provide a safe CUDA assert macro if not already defined */
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
static float sigma_last = -10.0;


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


/* ---------------------------
   Device kernels
   Each thread uses a linear global thread id and processes pixels
   in a linear loop so the grid dimension limits are not a problem.
   --------------------------- */


/* Horizontal convolution*/
__global__ static void convolve_horiz_kernel(
  const float *imgin,
  float *imgout,
  int ncols,
  int nrows,
  const float *kernel, 
  int kwidth,
  int radius)
{
  const size_t npixels = (size_t)ncols * (size_t)nrows;

  // computes a unique linear thread id
  size_t threads_per_block = (size_t)blockDim.x * (size_t)blockDim.y;
  size_t blocks_total = (size_t)gridDim.x * (size_t)gridDim.y;
  size_t tid_in_block = (size_t)threadIdx.y * (size_t)blockDim.x + (size_t)threadIdx.x;
  size_t block_index = (size_t)blockIdx.y * (size_t)gridDim.x + (size_t)blockIdx.x;
  size_t tid = block_index * threads_per_block + tid_in_block;

  size_t stride = blocks_total * threads_per_block;

  for (size_t idx = tid; idx < npixels; idx += stride) {
    int row = (int)(idx / ncols);
    int col = (int)(idx % ncols);

    // left and right border handling 
    if (col < radius || col >= (ncols - radius)) {
      imgout[idx] = 0.0f;
      continue;
    }

    float sum = 0.0f;
    int pbase = row * ncols + (col - radius);
    for (int k = 0; k < kwidth; ++k) {
      float v = imgin[pbase + k];
      float kv = kernel[kwidth - 1 - k];
      sum += v * kv;
    }
    imgout[idx] = sum;
  }
}

/* Vertical convolution*/
__global__ static void convolve_vert_kernel(
  const float *imgin,
  float *imgout,
  int ncols,
  int nrows,
  const float *kernel,
  int kwidth,
  int radius)
{
  const size_t npixels = (size_t)ncols * (size_t)nrows;

  // computes a unique linear thread id
  size_t threads_per_block = (size_t)blockDim.x * (size_t)blockDim.y;
  size_t blocks_total = (size_t)gridDim.x * (size_t)gridDim.y;
  size_t tid_in_block = (size_t)threadIdx.y * (size_t)blockDim.x + (size_t)threadIdx.x;
  size_t block_index = (size_t)blockIdx.y * (size_t)gridDim.x + (size_t)blockIdx.x;
  size_t tid = block_index * threads_per_block + tid_in_block;

  size_t stride = blocks_total * threads_per_block;

  for (size_t idx = tid; idx < npixels; idx += stride) {
    int row = (int)(idx / ncols);
    int col = (int)(idx % ncols);

    if (row < radius || row >= (nrows - radius)) {
      imgout[idx] = 0.0f;
      continue;
    }

    float sum = 0.0f;
    int pbase = (row - radius) * ncols + col;
    for (int k = 0; k < kwidth; ++k) {
      float v = imgin[pbase + k * ncols];
      float kv = kernel[kwidth - 1 - k];
      sum += v * kv;
    }
    imgout[idx] = sum;
  }
}


/*********************************************************************
 * _convolveImageHoriz
 */
extern "C" void convolveImageHoriz_cuda(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  /* checks*/
  assert(kernel.width % 2 == 1);
  assert(imgin != imgout);
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  int ncols = imgin->ncols;
  int nrows = imgin->nrows;
  size_t npixels = (size_t)ncols * (size_t)nrows;
  size_t bytes = npixels * sizeof(float);
  int kwidth = kernel.width;
  int radius = kwidth / 2;
  size_t kernel_bytes = (size_t)kwidth * sizeof(float);

  float *d_in = nullptr;
  float *d_out = nullptr;
  float *d_kernel = nullptr;

  // allocate device memory
  KLT_CUDA_ASSERT(cudaMalloc((void**)&d_in, bytes));
  KLT_CUDA_ASSERT(cudaMalloc((void**)&d_out, bytes));
  KLT_CUDA_ASSERT(cudaMalloc((void**)&d_kernel, kernel_bytes));

  // copy Host to Device
  KLT_CUDA_ASSERT(cudaMemcpy(d_in, imgin->data, bytes, cudaMemcpyHostToDevice));
  KLT_CUDA_ASSERT(cudaMemcpy(d_kernel, kernel.data, kernel_bytes, cudaMemcpyHostToDevice));


  const int TX = 32;
  const int TY = 8; 
  dim3 block(TX, TY);
 
  dim3 grid((ncols + TX - 1) / TX, (nrows + TY - 1) / TY);

  // launch
  convolve_horiz_kernel<<<grid, block>>>(d_in, d_out, ncols, nrows, d_kernel, kwidth, radius);

  // error check
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "convolve_horiz_kernel launch failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_kernel);
    abort();
  }
  KLT_CUDA_ASSERT(cudaDeviceSynchronize());

  // copy back
  KLT_CUDA_ASSERT(cudaMemcpy(imgout->data, d_out, bytes, cudaMemcpyDeviceToHost));

  // free
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_kernel);
}

extern "C" void convolveImageVert_cuda(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  /*  checks (same as CPU code) */

  assert(kernel.width % 2 == 1);
  assert(imgin != imgout);
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  int ncols = imgin->ncols;
  int nrows = imgin->nrows;
  size_t npixels = (size_t)ncols * (size_t)nrows;
  size_t bytes = npixels * sizeof(float);
  int kwidth = kernel.width;
  int radius = kwidth / 2;
  size_t kernel_bytes = (size_t)kwidth * sizeof(float);

  float *d_in = nullptr;
  float *d_out = nullptr;
  float *d_kernel = nullptr;

  // allocate
  KLT_CUDA_ASSERT(cudaMalloc((void**)&d_in, bytes));
  KLT_CUDA_ASSERT(cudaMalloc((void**)&d_out, bytes));
  KLT_CUDA_ASSERT(cudaMalloc((void**)&d_kernel, kernel_bytes));

  // copy
  KLT_CUDA_ASSERT(cudaMemcpy(d_in, imgin->data, bytes, cudaMemcpyHostToDevice));
  KLT_CUDA_ASSERT(cudaMemcpy(d_kernel, kernel.data, kernel_bytes, cudaMemcpyHostToDevice));

  
  const int TX = 32; 
  const int TY = 8;  
  dim3 block(TX, TY);
  dim3 grid((ncols + TX - 1) / TX, (nrows + TY - 1) / TY);

  
  cudaEvent_t start, stop;
  KLT_CUDA_ASSERT(cudaEventCreate(&start));
  KLT_CUDA_ASSERT(cudaEventCreate(&stop));
  KLT_CUDA_ASSERT(cudaEventRecord(start));

  convolve_vert_kernel<<<grid, block>>>(d_in, d_out, ncols, nrows, d_kernel, kwidth, radius);

  // check error
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "convolve_vert_kernel launch failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_kernel);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    abort();
  }

  KLT_CUDA_ASSERT(cudaEventRecord(stop));
  KLT_CUDA_ASSERT(cudaEventSynchronize(stop));

  float milliseconds = 0.0f;
  KLT_CUDA_ASSERT(cudaEventElapsedTime(&milliseconds, start, stop));
  // For debug timings, uncomment this:
  // printf("convolve_vert_kernel time: %.3f ms\n", milliseconds);

  KLT_CUDA_ASSERT(cudaEventDestroy(start));
  KLT_CUDA_ASSERT(cudaEventDestroy(stop));

  KLT_CUDA_ASSERT(cudaDeviceSynchronize());

  // copy back
  KLT_CUDA_ASSERT(cudaMemcpy(imgout->data, d_out, bytes, cudaMemcpyDeviceToHost));

  // free
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_kernel);
}

/*********************************************************************
 * _convolveSeparate
 */

static void _convolveSeparate(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
  _KLT_FloatImage tmpimg;
  tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);


  cudaEvent_t start, end;
  float elapsed_ms = 0.0f;


  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  convolveImageHoriz_cuda(imgin, horiz_kernel, tmpimg);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_ms, start, end);
  printf("Time for convolveImageHoriz_cuda: %.3f ms\n", elapsed_ms);

 -
  cudaEventRecord(start, 0);

  convolveImageVert_cuda(tmpimg, vert_kernel, imgout);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_ms, start, end);
  printf("Time for convolveImageVert_cuda: %.3f ms\n", elapsed_ms);

 
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  _KLTFreeFloatImage(tmpimg);
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
