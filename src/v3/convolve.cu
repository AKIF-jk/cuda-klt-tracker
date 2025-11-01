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
  /* No device copy here; device IO is handled inside GPU convolution functions */
  
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


// /*********************************************************************
//  * _convolveImageHoriz
//  */
// extern "C" void convolveImageHoriz_cuda(
//   _KLT_FloatImage imgin,
//   ConvolutionKernel kernel,
//   _KLT_FloatImage imgout)
// {
//   /* checks*/
//   assert(kernel.width % 2 == 1);
//   assert(imgin != imgout);
//   assert(imgout->ncols >= imgin->ncols);
//   assert(imgout->nrows >= imgin->nrows);

//   int ncols = imgin->ncols;
//   int nrows = imgin->nrows;
//   size_t npixels = (size_t)ncols * (size_t)nrows;
//   size_t bytes = npixels * sizeof(float);
//   int kwidth = kernel.width;
//   int radius = kwidth / 2;
//   size_t kernel_bytes = (size_t)kwidth * sizeof(float);

//   float *d_in = nullptr;
//   float *d_out = nullptr;
//   float *d_kernel = nullptr;
//   // allocate device memory
//  //  KLT_CUDA_ASSERT(cudaMalloc((void**)&d_in, bytes));
//   // KLT_CUDA_ASSERT(cudaMalloc((void**)&d_out, bytes));
//   KLT_CUDA_ASSERT(cudaMalloc((void**)&d_kernel, kernel_bytes));

//   // copy Host to Device
//    //KLT_CUDA_ASSERT(cudaMemcpy(d_in, imgin->data, bytes, cudaMemcpyHostToDevice));
//   KLT_CUDA_ASSERT(cudaMemcpy(d_kernel, kernel.data, kernel_bytes, cudaMemcpyHostToDevice));


//   const int TX = 32;
//   const int TY = 8; 
//   dim3 block(TX, TY);
 
//   dim3 grid((ncols + TX - 1) / TX, (nrows + TY - 1) / TY);
//   // launch
//   if(imgin->device_data==NULL) printf("NULL\n");
//   else printf("NOT NULL\n");
 

//   convolve_horiz_kernel<<<grid, block>>>(imgin->device_data, imgout->device_data, ncols, nrows, d_kernel, kwidth, radius);
//   //convolve_horiz_kernel<<<grid, block>>>(d_in,d_out, ncols, nrows, d_kernel, kwidth, radius);

//   // error check
//   cudaError_t err = cudaGetLastError();
//   if (err != cudaSuccess) {
//     fprintf(stderr, "convolve_horiz_kernel launch failed: %s\n", cudaGetErrorString(err));
//     // cudaFree(d_in); cudaFree(d_out); cudaFree(d_kernel);
//     abort();
//   }
//   KLT_CUDA_ASSERT(cudaDeviceSynchronize());

//   // copy back
//   KLT_CUDA_ASSERT(cudaMemcpy(imgout->data,imgout->device_data, bytes, cudaMemcpyDeviceToHost));

//   // free
//   // cudaFree(d_in);
//   // cudaFree(d_out);
//   cudaFree(d_kernel);
// }

// extern "C" void convolveImageVert_cuda(
//   _KLT_FloatImage imgin,
//   ConvolutionKernel kernel,
//   _KLT_FloatImage imgout)
// {
//   /*  checks (same as CPU code) */

//   assert(kernel.width % 2 == 1);
//   assert(imgin != imgout);
//   assert(imgout->ncols >= imgin->ncols);
//   assert(imgout->nrows >= imgin->nrows);

//   int ncols = imgin->ncols;
//   int nrows = imgin->nrows;
//   size_t npixels = (size_t)ncols * (size_t)nrows;
//   size_t bytes = npixels * sizeof(float);
//   int kwidth = kernel.width;
//   int radius = kwidth / 2;
//   size_t kernel_bytes = (size_t)kwidth * sizeof(float);

//    float *d_in = nullptr;
//    float *d_out = nullptr;
//   float *d_kernel = nullptr;

//   // allocate
//   //KLT_CUDA_ASSERT(cudaMalloc((void**)&d_in, bytes));
//   //KLT_CUDA_ASSERT(cudaMalloc((void**)&d_out, bytes));
//   KLT_CUDA_ASSERT(cudaMalloc((void**)&d_kernel, kernel_bytes));

//   // copy
//    //KLT_CUDA_ASSERT(cudaMemcpy(d_in, imgin->data, bytes, cudaMemcpyHostToDevice));
//   KLT_CUDA_ASSERT(cudaMemcpy(d_kernel, kernel.data, kernel_bytes, cudaMemcpyHostToDevice));

  
//   const int TX = 32; 
//   const int TY = 8;  
//   dim3 block(TX, TY);
//   dim3 grid((ncols + TX - 1) / TX, (nrows + TY - 1) / TY);

  
//   cudaEvent_t start, stop;
//   KLT_CUDA_ASSERT(cudaEventCreate(&start));
//   KLT_CUDA_ASSERT(cudaEventCreate(&stop));
//   KLT_CUDA_ASSERT(cudaEventRecord(start));

//   if(imgin->device_data==NULL) printf("NULL\n");
  
//   convolve_vert_kernel<<<grid, block>>>(imgin->device_data, imgout->device_data, ncols, nrows, d_kernel, kwidth, radius);

//   // check error
//   cudaError_t err = cudaGetLastError();
//   if (err != cudaSuccess) {
//     fprintf(stderr, "convolve_vert_kernel launch failed: %s\n", cudaGetErrorString(err));
//     // cudaFree(d_in); cudaFree(d_out); cudaFree(d_kernel);
//     cudaEventDestroy(start); cudaEventDestroy(stop);
//     abort();
//   }

//   KLT_CUDA_ASSERT(cudaEventRecord(stop));
//   KLT_CUDA_ASSERT(cudaEventSynchronize(stop));

//   float milliseconds = 0.0f;
//   KLT_CUDA_ASSERT(cudaEventElapsedTime(&milliseconds, start, stop));
//   // For debug timings, uncomment this:
//   // printf("convolve_vert_kernel time: %.3f ms\n", milliseconds);

//   KLT_CUDA_ASSERT(cudaEventDestroy(start));
//   KLT_CUDA_ASSERT(cudaEventDestroy(stop));

//   KLT_CUDA_ASSERT(cudaDeviceSynchronize());

//   // copy back
//   KLT_CUDA_ASSERT(cudaMemcpy(imgout->data, imgout->device_data, bytes, cudaMemcpyDeviceToHost));

//   // free
//   // cudaFree(d_in);
//   // cudaFree(d_out);
//   cudaFree(d_kernel);
// }

/*********************************************************************
 * _convolveSeparate
 */

static void _convolveSeparate(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
  /* Keep data resident on device across both passes; one H2D and one D2H per call */
  const int ncols = imgin->ncols;
  const int nrows = imgin->nrows;
  const size_t npixels = (size_t)ncols * (size_t)nrows;
  const size_t bytes = npixels * sizeof(float);

  static float *d_in = NULL;
  static float *d_tmp = NULL;
  static float *d_out = NULL;
  static size_t cap_bytes = 0;

  if (bytes > cap_bytes) {
    if (d_in) { cudaFree(d_in); d_in = NULL; }
    if (d_tmp) { cudaFree(d_tmp); d_tmp = NULL; }
    if (d_out) { cudaFree(d_out); d_out = NULL; }
    KLT_CUDA_ASSERT(cudaMalloc((void**)&d_in, bytes));
    KLT_CUDA_ASSERT(cudaMalloc((void**)&d_tmp, bytes));
    KLT_CUDA_ASSERT(cudaMalloc((void**)&d_out, bytes));
    cap_bytes = bytes;
  }

  /* Always upload current frame; host buffers may be reused with new contents across frames */
  KLT_CUDA_ASSERT(cudaMemcpy(d_in, imgin->data, bytes, cudaMemcpyHostToDevice));

  float *d_kernel = NULL;
  KLT_CUDA_ASSERT(cudaMalloc((void**)&d_kernel, (size_t)MAX_KERNEL_WIDTH * sizeof(float)));

  const int TX = 32;
  const int TY = 8;
  dim3 block(TX, TY);
  dim3 grid((ncols + TX - 1) / TX, (nrows + TY - 1) / TY);

  cudaEvent_t start, end;
  float elapsed_ms = 0.0f;
  KLT_CUDA_ASSERT(cudaEventCreate(&start));
  KLT_CUDA_ASSERT(cudaEventCreate(&end));

  /* Horizontal pass */
  KLT_CUDA_ASSERT(cudaMemcpy(d_kernel, horiz_kernel.data, (size_t)horiz_kernel.width * sizeof(float), cudaMemcpyHostToDevice));
  KLT_CUDA_ASSERT(cudaEventRecord(start));
  convolve_horiz_kernel<<<grid, block>>>(d_in, d_tmp, ncols, nrows, d_kernel, horiz_kernel.width, horiz_kernel.width / 2);
  KLT_CUDA_ASSERT(cudaGetLastError());
  KLT_CUDA_ASSERT(cudaEventRecord(end));
  KLT_CUDA_ASSERT(cudaEventSynchronize(end));
  KLT_CUDA_ASSERT(cudaEventElapsedTime(&elapsed_ms, start, end));
  printf("Time for convolveImageHoriz_cuda: %.3f ms\n", elapsed_ms);

  /* Vertical pass */
  KLT_CUDA_ASSERT(cudaMemcpy(d_kernel, vert_kernel.data, (size_t)vert_kernel.width * sizeof(float), cudaMemcpyHostToDevice));
  KLT_CUDA_ASSERT(cudaEventRecord(start));
  convolve_vert_kernel<<<grid, block>>>(d_tmp, d_out, ncols, nrows, d_kernel, vert_kernel.width, vert_kernel.width / 2);
  KLT_CUDA_ASSERT(cudaGetLastError());
  KLT_CUDA_ASSERT(cudaEventRecord(end));
  KLT_CUDA_ASSERT(cudaEventSynchronize(end));
  KLT_CUDA_ASSERT(cudaEventElapsedTime(&elapsed_ms, start, end));
  printf("Time for convolveImageVert_cuda: %.3f ms\n", elapsed_ms);

  KLT_CUDA_ASSERT(cudaMemcpy(imgout->data, d_out, bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_kernel);
  KLT_CUDA_ASSERT(cudaEventDestroy(start));
  KLT_CUDA_ASSERT(cudaEventDestroy(end));
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

// /* Standard includes */
// #include <assert.h>
// #include <math.h>
// #include <stdlib.h>
// #include <stdio.h>

// /* Our includes */
// #include "base.h"
// #include "error.h"
// #include "convolve.h"
// #include "klt_util.h"
// #include <cuda_runtime.h>

// #define MAX_KERNEL_WIDTH 	71

// #ifndef KLT_CUDA_ASSERT
// #define KLT_CUDA_ASSERT(call)                                                  \
// do {                                                                           \
//     cudaError_t err = (call);                                                  \
//     if (err != cudaSuccess) {                                                  \
//         fprintf(stderr, "CUDA error at %s:%d (%s): %s\n",                      \
//                 __FILE__, __LINE__, #call, cudaGetErrorString(err));          \
//         fflush(stderr);                                                        \
//         abort();                                                               \
//     }                                                                          \
// } while (0)
// #endif

// typedef struct  {
//   int width;
//   float data[MAX_KERNEL_WIDTH];
// }  ConvolutionKernel;

// /* Kernels */
// static ConvolutionKernel gauss_kernel;
// static ConvolutionKernel gaussderiv_kernel;
// static float sigma_last = -10.0f;

// /* Constant memory for active kernel (updated per pass) */
// __constant__ float cKernel[MAX_KERNEL_WIDTH];

// /*********************************************************************
//  * _KLTToFloatImage
//  */
// void _KLTToFloatImage(
//   KLT_PixelType *img,
//   int ncols, int nrows,
//   _KLT_FloatImage floatimg)
// {
//   KLT_PixelType *ptrend = img + ncols*nrows;
//   float *ptrout = floatimg->data;

//   assert(floatimg->ncols >= ncols);
//   assert(floatimg->nrows >= nrows);

//   floatimg->ncols = ncols;
//   floatimg->nrows = nrows;

//   while (img < ptrend)  *ptrout++ = (float) *img++;
// }

// /*********************************************************************
//  * _computeKernels (same as CPU reference)
//  */
// static void _computeKernels(
//   float sigma,
//   ConvolutionKernel *gauss,
//   ConvolutionKernel *gaussderiv)
// {
//   const float factor = 0.01f;   /* for truncating tail */
//   int i;

//   assert(MAX_KERNEL_WIDTH % 2 == 1);
//   assert(sigma >= 0.0);

//   {
//     const int hw = MAX_KERNEL_WIDTH / 2;
//     float max_gauss = 1.0f, max_gaussderiv = (float) (sigma*expf(-0.5f));
//     for (i = -hw ; i <= hw ; i++)  {
//       gauss->data[i+hw]      = (float) expf(-i*i / (2*sigma*sigma));
//       gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
//     }
//     gauss->width = MAX_KERNEL_WIDTH;
//     for (i = -hw ; fabsf(gauss->data[i+hw] / max_gauss) < factor ; i++, gauss->width -= 2);
//     gaussderiv->width = MAX_KERNEL_WIDTH;
//     for (i = -hw ; fabsf(gaussderiv->data[i+hw] / max_gaussderiv) < factor ; i++, gaussderiv->width -= 2);
//     if (gauss->width == MAX_KERNEL_WIDTH || gaussderiv->width == MAX_KERNEL_WIDTH)
//       KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for a sigma of %f", MAX_KERNEL_WIDTH, sigma);
//   }

//   for (i = 0 ; i < gauss->width ; i++)
//     gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
//   for (i = 0 ; i < gaussderiv->width ; i++)
//     gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];
//   {
//     const int hw = gaussderiv->width / 2;
//     float den = 0.0f;
//     for (i = 0 ; i < gauss->width ; i++)  den += gauss->data[i];
//     for (i = 0 ; i < gauss->width ; i++)  gauss->data[i] /= den;
//     den = 0.0f;
//     for (i = -hw ; i <= hw ; i++)  den -= i*gaussderiv->data[i+hw];
//     for (i = -hw ; i <= hw ; i++)  gaussderiv->data[i+hw] /= den;
//   }

//   sigma_last = sigma;
// }

// void _KLTGetKernelWidths(
//   float sigma,
//   int *gauss_width,
//   int *gaussderiv_width)
// {
//   _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
//   *gauss_width = gauss_kernel.width;
//   *gaussderiv_width = gaussderiv_kernel.width;
// }

// /*********************************************************************
//  * Device kernels: shared-memory tiled separable convolution
//  *********************************************************************/

// /* Horizontal: tile size (TX x TY), with halo of radius R on X */
// template<int TX, int TY>
// __global__ void convolve_horiz_tiled(
//   const float* __restrict__ imgin,
//   float* __restrict__ imgout,
//   int ncols,
//   int nrows,
//   int kwidth,
//   int radius)
// {
//   extern __shared__ float smem[]; // size = TY * (TX + 2*radius)
//   const int x = blockIdx.x * TX + threadIdx.x;
//   const int y = blockIdx.y * TY + threadIdx.y;

//   if (y >= nrows) return;

//   const int tileW = TX + 2*radius;
//   const int lane = threadIdx.y * tileW + threadIdx.x; // for coalesced stores

//   // Global start index for this tile row
//   const int rowStart = y * ncols;

//   // Load main region
//   if (x < ncols) {
//     smem[lane + radius] = __ldg(&imgin[rowStart + x]);
//   }

//   // Load left halo
//   for (int hx = threadIdx.x; hx < radius; hx += TX) {
//     int gx = blockIdx.x * TX + hx - radius;
//     float v = 0.0f;
//     if (gx >= 0 && gx < ncols) v = __ldg(&imgin[rowStart + gx]);
//     smem[threadIdx.y * tileW + hx] = v;
//   }
//   // Load right halo
//   for (int hx = threadIdx.x; hx < radius; hx += TX) {
//     int gx = blockIdx.x * TX + TX + hx;
//     float v = 0.0f;
//     if (gx >= 0 && gx < ncols) v = __ldg(&imgin[rowStart + gx]);
//     smem[threadIdx.y * tileW + radius + TX + hx] = v;
//   }

//   __syncthreads();

//   if (x >= ncols) return;

//   // Border policy: zero out borders consistent with CPU
//   if (x < radius || x >= (ncols - radius)) {
//     imgout[rowStart + x] = 0.0f;
//     return;
//   }

//   float sum = 0.0f;
//   const int base = threadIdx.y * tileW + threadIdx.x; // position including halo offset when adding k
//   // Reverse-order multiply like reference
//   for (int k = 0; k < kwidth; ++k) {
//     float v = smem[base + k];
//     float kv = cKernel[kwidth - 1 - k];
//     sum += v * kv;
//   }
//   imgout[rowStart + x] = sum;
// }

// /* Vertical: tile size (TX x TY), with halo of radius R on Y */
// template<int TX, int TY>
// __global__ void convolve_vert_tiled(
//   const float* __restrict__ imgin,
//   float* __restrict__ imgout,
//   int ncols,
//   int nrows,
//   int kwidth,
//   int radius)
// {
//   extern __shared__ float smem[]; // size = (TY + 2*radius) * TX
//   const int x = blockIdx.x * TX + threadIdx.x;
//   const int y = blockIdx.y * TY + threadIdx.y;

//   if (x >= ncols) return;

//   const int tileH = TY + 2*radius;
//   const int lane = (threadIdx.y + radius) * TX + threadIdx.x;

//   // Load main region
//   if (y < nrows) {
//     smem[lane] = __ldg(&imgin[y * ncols + x]);
//   }

//   // Load top halo
//   for (int hy = threadIdx.y; hy < radius; hy += TY) {
//     int gy = blockIdx.y * TY + hy - radius;
//     float v = 0.0f;
//     if (gy >= 0 && gy < nrows) v = __ldg(&imgin[gy * ncols + x]);
//     smem[hy * TX + threadIdx.x] = v;
//   }
//   // Load bottom halo
//   for (int hy = threadIdx.y; hy < radius; hy += TY) {
//     int gy = blockIdx.y * TY + TY + hy;
//     float v = 0.0f;
//     if (gy >= 0 && gy < nrows) v = __ldg(&imgin[gy * ncols + x]);
//     smem[(radius + TY + hy) * TX + threadIdx.x] = v;
//   }

//   __syncthreads();

//   if (y >= nrows) return;

//   if (y < radius || y >= (nrows - radius)) {
//     imgout[y * ncols + x] = 0.0f;
//     return;
//   }

//   float sum = 0.0f;
//   const int base = threadIdx.x; // advance by TX per step
//   int smem_row = (threadIdx.y) * TX; // starting row aligned so that adding k walks down
//   for (int k = 0; k < kwidth; ++k) {
//     float v = smem[(smem_row + k * TX) + base];
//     float kv = cKernel[kwidth - 1 - k];
//     sum += v * kv;
//   }
//   imgout[y * ncols + x] = sum;
// }

// /*********************************************************************
//  * Host helpers
//  *********************************************************************/

// static void launch_horizontal(
//   const float* d_in,
//   float* d_out,
//   int ncols,
//   int nrows,
//   const ConvolutionKernel& kernel,
//   cudaStream_t stream)
// {
//   const int kwidth = kernel.width;
//   const int radius = kwidth / 2;
//   KLT_CUDA_ASSERT(cudaMemcpyToSymbolAsync(cKernel, kernel.data, sizeof(float) * kwidth, 0, cudaMemcpyHostToDevice, stream));

//   const int TX = 32;
//   const int TY = 8;
//   dim3 block(TX, TY);
//   dim3 grid((ncols + TX - 1) / TX, (nrows + TY - 1) / TY);
//   size_t smem_bytes = (size_t)TY * (size_t)(TX + 2*radius) * sizeof(float);
//   convolve_horiz_tiled<TX, TY><<<grid, block, smem_bytes, stream>>>(d_in, d_out, ncols, nrows, kwidth, radius);
//   KLT_CUDA_ASSERT(cudaGetLastError());
// }

// static void launch_vertical(
//   const float* d_in,
//   float* d_out,
//   int ncols,
//   int nrows,
//   const ConvolutionKernel& kernel,
//   cudaStream_t stream)
// {
//   const int kwidth = kernel.width;
//   const int radius = kwidth / 2;
//   KLT_CUDA_ASSERT(cudaMemcpyToSymbolAsync(cKernel, kernel.data, sizeof(float) * kwidth, 0, cudaMemcpyHostToDevice, stream));

//   const int TX = 32;
//   const int TY = 8;
//   dim3 block(TX, TY);
//   dim3 grid((ncols + TX - 1) / TX, (nrows + TY - 1) / TY);
//   size_t smem_bytes = (size_t)(TY + 2*radius) * (size_t)TX * sizeof(float);
//   convolve_vert_tiled<TX, TY><<<grid, block, smem_bytes, stream>>>(d_in, d_out, ncols, nrows, kwidth, radius);
//   KLT_CUDA_ASSERT(cudaGetLastError());
// }

// /*********************************************************************
//  * Optimized _convolveSeparate using on-device intermediate
//  *********************************************************************/
// static void _convolveSeparate(
//   _KLT_FloatImage imgin,
//   ConvolutionKernel horiz_kernel,
//   ConvolutionKernel vert_kernel,
//   _KLT_FloatImage imgout)
// {
//   const int ncols = imgin->ncols;
//   const int nrows = imgin->nrows;
//   const size_t npixels = (size_t)ncols * (size_t)nrows;
//   const size_t bytes = npixels * sizeof(float);

//   assert(imgout->ncols >= ncols);
//   assert(imgout->nrows >= nrows);

//   float *d_in = NULL, *d_tmp = NULL, *d_out = NULL;
//   cudaStream_t stream;
//   KLT_CUDA_ASSERT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
//   KLT_CUDA_ASSERT(cudaMalloc((void**)&d_in, bytes));
//   KLT_CUDA_ASSERT(cudaMalloc((void**)&d_tmp, bytes));
//   KLT_CUDA_ASSERT(cudaMalloc((void**)&d_out, bytes));

//   // H2D once
//   KLT_CUDA_ASSERT(cudaMemcpyAsync(d_in, imgin->data, bytes, cudaMemcpyHostToDevice, stream));

//   // Horizontal pass: d_in -> d_tmp
//   launch_horizontal(d_in, d_tmp, ncols, nrows, horiz_kernel, stream);

//   // Vertical pass: d_tmp -> d_out
//   launch_vertical(d_tmp, d_out, ncols, nrows, vert_kernel, stream);

//   // D2H once
//   KLT_CUDA_ASSERT(cudaMemcpyAsync(imgout->data, d_out, bytes, cudaMemcpyDeviceToHost, stream));
//   KLT_CUDA_ASSERT(cudaStreamSynchronize(stream));

//   cudaFree(d_in);
//   cudaFree(d_tmp);
//   cudaFree(d_out);
//   cudaStreamDestroy(stream);
// }

// /*********************************************************************
//  * Public APIs mirroring CPU
//  *********************************************************************/
// void _KLTComputeGradients(
//   _KLT_FloatImage img,
//   float sigma,
//   _KLT_FloatImage gradx,
//   _KLT_FloatImage grady)
// {
//   assert(gradx->ncols >= img->ncols);
//   assert(gradx->nrows >= img->nrows);
//   assert(grady->ncols >= img->ncols);
//   assert(grady->nrows >= img->nrows);

//   if (fabsf(sigma - sigma_last) > 0.05f)
//     _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

//   _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
//   _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);
// }

// void _KLTComputeSmoothedImage(
//   _KLT_FloatImage img,
//   float sigma,
//   _KLT_FloatImage smooth)
// {
//   assert(smooth->ncols >= img->ncols);
//   assert(smooth->nrows >= img->nrows);

//   if (fabsf(sigma - sigma_last) > 0.05f)
//     _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

//   _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
// }

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

// /* Standard includes */
// #include <assert.h>
// #include <math.h>
// #include <stdlib.h>
// #include <stdio.h>

// /* Our includes */
// #include "base.h"
// #include "error.h"
// #include "convolve.h"
// #include "klt_util.h"
// #include <cuda_runtime.h>

// #define MAX_KERNEL_WIDTH 	71

// #ifndef KLT_CUDA_ASSERT
// #define KLT_CUDA_ASSERT(call)                                                  \
// do {                                                                           \
//     cudaError_t err = (call);                                                  \
//     if (err != cudaSuccess) {                                                  \
//         fprintf(stderr, "CUDA error at %s:%d (%s): %s\n",                      \
//                 __FILE__, __LINE__, #call, cudaGetErrorString(err));          \
//         fflush(stderr);                                                        \
//         abort();                                                               \
//     }                                                                          \
// } while (0)
// #endif

// typedef struct  {
//   int width;
//   float data[MAX_KERNEL_WIDTH];
// }  ConvolutionKernel;

// /* Kernels */
// static ConvolutionKernel gauss_kernel;
// static ConvolutionKernel gaussderiv_kernel;
// static float sigma_last = -10.0f;

// /* Constant memory for active kernel (updated per pass) */
// __constant__ float cKernel[MAX_KERNEL_WIDTH];

// /* Persistent CUDA context to avoid per-call alloc/free and enable reuse */
// typedef struct {
//   float *d_in;
//   float *d_tmp;
//   float *d_out;
//   size_t cap_bytes;
//   cudaStream_t stream;
// } ConvGPUCtx;

// static ConvGPUCtx g_ctx = { nullptr, nullptr, nullptr, 0, nullptr };

// static void ensure_capacity(size_t bytes)
// {
//   if (g_ctx.stream == nullptr) {
//     KLT_CUDA_ASSERT(cudaStreamCreateWithFlags(&g_ctx.stream, cudaStreamNonBlocking));
//   }
//   if (bytes <= g_ctx.cap_bytes) return;
//   if (g_ctx.d_in) { cudaFree(g_ctx.d_in); g_ctx.d_in = nullptr; }
//   if (g_ctx.d_tmp) { cudaFree(g_ctx.d_tmp); g_ctx.d_tmp = nullptr; }
//   if (g_ctx.d_out) { cudaFree(g_ctx.d_out); g_ctx.d_out = nullptr; }
//   KLT_CUDA_ASSERT(cudaMalloc((void**)&g_ctx.d_in, bytes));
//   KLT_CUDA_ASSERT(cudaMalloc((void**)&g_ctx.d_tmp, bytes));
//   KLT_CUDA_ASSERT(cudaMalloc((void**)&g_ctx.d_out, bytes));
//   g_ctx.cap_bytes = bytes;
// }

// void _KLTReleaseGPU()
// {
//   if (g_ctx.d_in) { cudaFree(g_ctx.d_in); g_ctx.d_in = nullptr; }
//   if (g_ctx.d_tmp) { cudaFree(g_ctx.d_tmp); g_ctx.d_tmp = nullptr; }
//   if (g_ctx.d_out) { cudaFree(g_ctx.d_out); g_ctx.d_out = nullptr; }
//   if (g_ctx.stream) { cudaStreamDestroy(g_ctx.stream); g_ctx.stream = nullptr; }
//   g_ctx.cap_bytes = 0;
// }

// /*********************************************************************
//  * _KLTToFloatImage
//  */
// void _KLTToFloatImage(
//   KLT_PixelType *img,
//   int ncols, int nrows,
//   _KLT_FloatImage floatimg)
// {
//   KLT_PixelType *ptrend = img + ncols*nrows;
//   float *ptrout = floatimg->data;

//   assert(floatimg->ncols >= ncols);
//   assert(floatimg->nrows >= nrows);

//   floatimg->ncols = ncols;
//   floatimg->nrows = nrows;

//   while (img < ptrend)  *ptrout++ = (float) *img++;
// }

// /*********************************************************************
//  * _computeKernels (same as CPU reference)
//  */
// static void _computeKernels(
//   float sigma,
//   ConvolutionKernel *gauss,
//   ConvolutionKernel *gaussderiv)
// {
//   const float factor = 0.01f;   /* for truncating tail */
//   int i;

//   assert(MAX_KERNEL_WIDTH % 2 == 1);
//   assert(sigma >= 0.0);

//   {
//     const int hw = MAX_KERNEL_WIDTH / 2;
//     float max_gauss = 1.0f, max_gaussderiv = (float) (sigma*expf(-0.5f));
//     for (i = -hw ; i <= hw ; i++)  {
//       gauss->data[i+hw]      = (float) expf(-i*i / (2*sigma*sigma));
//       gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
//     }
//     gauss->width = MAX_KERNEL_WIDTH;
//     for (i = -hw ; fabsf(gauss->data[i+hw] / max_gauss) < factor ; i++, gauss->width -= 2);
//     gaussderiv->width = MAX_KERNEL_WIDTH;
//     for (i = -hw ; fabsf(gaussderiv->data[i+hw] / max_gaussderiv) < factor ; i++, gaussderiv->width -= 2);
//     if (gauss->width == MAX_KERNEL_WIDTH || gaussderiv->width == MAX_KERNEL_WIDTH)
//       KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for a sigma of %f", MAX_KERNEL_WIDTH, sigma);
//   }

//   for (i = 0 ; i < gauss->width ; i++)
//     gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
//   for (i = 0 ; i < gaussderiv->width ; i++)
//     gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];
//   {
//     const int hw = gaussderiv->width / 2;
//     float den = 0.0f;
//     for (i = 0 ; i < gauss->width ; i++)  den += gauss->data[i];
//     for (i = 0 ; i < gauss->width ; i++)  gauss->data[i] /= den;
//     den = 0.0f;
//     for (i = -hw ; i <= hw ; i++)  den -= i*gaussderiv->data[i+hw];
//     for (i = -hw ; i <= hw ; i++)  gaussderiv->data[i+hw] /= den;
//   }

//   sigma_last = sigma;
// }

// void _KLTGetKernelWidths(
//   float sigma,
//   int *gauss_width,
//   int *gaussderiv_width)
// {
//   _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
//   *gauss_width = gauss_kernel.width;
//   *gaussderiv_width = gaussderiv_kernel.width;
// }

// /*********************************************************************
//  * Device kernels: shared-memory tiled separable convolution
//  *********************************************************************/

// /* Horizontal: tile size (TX x TY), with halo of radius R on X */
// template<int TX, int TY>
// __global__ void convolve_horiz_tiled(
//   const float* __restrict__ imgin,
//   float* __restrict__ imgout,
//   int ncols,
//   int nrows,
//   int kwidth,
//   int radius)
// {
//   extern __shared__ float smem[]; // size = TY * (TX + 2*radius)
//   const int x = blockIdx.x * TX + threadIdx.x;
//   const int y = blockIdx.y * TY + threadIdx.y;

//   if (y >= nrows) return;

//   const int tileW = TX + 2*radius;
//   const int lane = threadIdx.y * tileW + threadIdx.x; // for coalesced stores

//   // Global start index for this tile row
//   const int rowStart = y * ncols;

//   // Load main region
//   if (x < ncols) {
//     smem[lane + radius] = __ldg(&imgin[rowStart + x]);
//   }

//   // Load left halo
//   for (int hx = threadIdx.x; hx < radius; hx += TX) {
//     int gx = blockIdx.x * TX + hx - radius;
//     float v = 0.0f;
//     if (gx >= 0 && gx < ncols) v = __ldg(&imgin[rowStart + gx]);
//     smem[threadIdx.y * tileW + hx] = v;
//   }
//   // Load right halo
//   for (int hx = threadIdx.x; hx < radius; hx += TX) {
//     int gx = blockIdx.x * TX + TX + hx;
//     float v = 0.0f;
//     if (gx >= 0 && gx < ncols) v = __ldg(&imgin[rowStart + gx]);
//     smem[threadIdx.y * tileW + radius + TX + hx] = v;
//   }

//   __syncthreads();

//   if (x >= ncols) return;

//   // Border policy: zero out borders consistent with CPU
//   if (x < radius || x >= (ncols - radius)) {
//     imgout[rowStart + x] = 0.0f;
//     return;
//   }

//   float sum = 0.0f;
//   const int base = threadIdx.y * tileW + threadIdx.x; // position including halo offset when adding k
//   // Reverse-order multiply like reference
//   for (int k = 0; k < kwidth; ++k) {
//     float v = smem[base + k];
//     float kv = cKernel[kwidth - 1 - k];
//     sum += v * kv;
//   }
//   imgout[rowStart + x] = sum;
// }

// /* Vertical: tile size (TX x TY), with halo of radius R on Y */
// template<int TX, int TY>
// __global__ void convolve_vert_tiled(
//   const float* __restrict__ imgin,
//   float* __restrict__ imgout,
//   int ncols,
//   int nrows,
//   int kwidth,
//   int radius)
// {
//   extern __shared__ float smem[]; // size = (TY + 2*radius) * TX
//   const int x = blockIdx.x * TX + threadIdx.x;
//   const int y = blockIdx.y * TY + threadIdx.y;

//   if (x >= ncols) return;

//   const int tileH = TY + 2*radius;
//   const int lane = (threadIdx.y + radius) * TX + threadIdx.x;

//   // Load main region
//   if (y < nrows) {
//     smem[lane] = __ldg(&imgin[y * ncols + x]);
//   }

//   // Load top halo
//   for (int hy = threadIdx.y; hy < radius; hy += TY) {
//     int gy = blockIdx.y * TY + hy - radius;
//     float v = 0.0f;
//     if (gy >= 0 && gy < nrows) v = __ldg(&imgin[gy * ncols + x]);
//     smem[hy * TX + threadIdx.x] = v;
//   }
//   // Load bottom halo
//   for (int hy = threadIdx.y; hy < radius; hy += TY) {
//     int gy = blockIdx.y * TY + TY + hy;
//     float v = 0.0f;
//     if (gy >= 0 && gy < nrows) v = __ldg(&imgin[gy * ncols + x]);
//     smem[(radius + TY + hy) * TX + threadIdx.x] = v;
//   }

//   __syncthreads();

//   if (y >= nrows) return;

//   if (y < radius || y >= (nrows - radius)) {
//     imgout[y * ncols + x] = 0.0f;
//     return;
//   }

//   float sum = 0.0f;
//   const int base = threadIdx.x; // advance by TX per step
//   int smem_row = (threadIdx.y) * TX; // starting row aligned so that adding k walks down
//   for (int k = 0; k < kwidth; ++k) {
//     float v = smem[(smem_row + k * TX) + base];
//     float kv = cKernel[kwidth - 1 - k];
//     sum += v * kv;
//   }
//   imgout[y * ncols + x] = sum;
// }

// /*********************************************************************
//  * Host helpers
//  *********************************************************************/

// static void launch_horizontal(
//   const float* d_in,
//   float* d_out,
//   int ncols,
//   int nrows,
//   const ConvolutionKernel& kernel,
//   cudaStream_t stream)
// {
//   const int kwidth = kernel.width;
//   const int radius = kwidth / 2;
//   KLT_CUDA_ASSERT(cudaMemcpyToSymbolAsync(cKernel, kernel.data, sizeof(float) * kwidth, 0, cudaMemcpyHostToDevice, stream));

//   const int TX = 32;
//   const int TY = 8;
//   dim3 block(TX, TY);
//   dim3 grid((ncols + TX - 1) / TX, (nrows + TY - 1) / TY);
//   size_t smem_bytes = (size_t)TY * (size_t)(TX + 2*radius) * sizeof(float);
//   convolve_horiz_tiled<TX, TY><<<grid, block, smem_bytes, stream>>>(d_in, d_out, ncols, nrows, kwidth, radius);
//   KLT_CUDA_ASSERT(cudaGetLastError());
// }

// static void launch_vertical(
//   const float* d_in,
//   float* d_out,
//   int ncols,
//   int nrows,
//   const ConvolutionKernel& kernel,
//   cudaStream_t stream)
// {
//   const int kwidth = kernel.width;
//   const int radius = kwidth / 2;
//   KLT_CUDA_ASSERT(cudaMemcpyToSymbolAsync(cKernel, kernel.data, sizeof(float) * kwidth, 0, cudaMemcpyHostToDevice, stream));

//   const int TX = 32;
//   const int TY = 8;
//   dim3 block(TX, TY);
//   dim3 grid((ncols + TX - 1) / TX, (nrows + TY - 1) / TY);
//   size_t smem_bytes = (size_t)(TY + 2*radius) * (size_t)TX * sizeof(float);
//   convolve_vert_tiled<TX, TY><<<grid, block, smem_bytes, stream>>>(d_in, d_out, ncols, nrows, kwidth, radius);
//   KLT_CUDA_ASSERT(cudaGetLastError());
// }

// /*********************************************************************
//  * Optimized _convolveSeparate using on-device intermediate
//  *********************************************************************/
// static void _convolveSeparate(
//   _KLT_FloatImage imgin,
//   ConvolutionKernel horiz_kernel,
//   ConvolutionKernel vert_kernel,
//   _KLT_FloatImage imgout)
// {
//   const int ncols = imgin->ncols;
//   const int nrows = imgin->nrows;
//   const size_t npixels = (size_t)ncols * (size_t)nrows;
//   const size_t bytes = npixels * sizeof(float);

//   assert(imgout->ncols >= ncols);
//   assert(imgout->nrows >= nrows);

//   ensure_capacity(bytes);

//   // H2D once
//   KLT_CUDA_ASSERT(cudaMemcpyAsync(g_ctx.d_in, imgin->data, bytes, cudaMemcpyHostToDevice, g_ctx.stream));

//   // Horizontal pass: d_in -> d_tmp
//   launch_horizontal(g_ctx.d_in, g_ctx.d_tmp, ncols, nrows, horiz_kernel, g_ctx.stream);

//   // Vertical pass: d_tmp -> d_out
//   launch_vertical(g_ctx.d_tmp, g_ctx.d_out, ncols, nrows, vert_kernel, g_ctx.stream);

//   // D2H once
//   KLT_CUDA_ASSERT(cudaMemcpyAsync(imgout->data, g_ctx.d_out, bytes, cudaMemcpyDeviceToHost, g_ctx.stream));
//   KLT_CUDA_ASSERT(cudaStreamSynchronize(g_ctx.stream));
// }

// /*********************************************************************
//  * Public APIs mirroring CPU
//  *********************************************************************/
// void _KLTComputeGradients(
//   _KLT_FloatImage img,
//   float sigma,
//   _KLT_FloatImage gradx,
//   _KLT_FloatImage grady)
// {
//   assert(gradx->ncols >= img->ncols);
//   assert(gradx->nrows >= img->nrows);
//   assert(grady->ncols >= img->ncols);
//   assert(grady->nrows >= img->nrows);

//   if (fabsf(sigma - sigma_last) > 0.05f)
//     _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

//   const int ncols = img->ncols;
//   const int nrows = img->nrows;
//   const size_t bytes = (size_t)ncols * (size_t)nrows * sizeof(float);

//   ensure_capacity(bytes);

//   // Upload image once
//   KLT_CUDA_ASSERT(cudaMemcpyAsync(g_ctx.d_in, img->data, bytes, cudaMemcpyHostToDevice, g_ctx.stream));

//   // gradx: (gaussderiv then gauss)
//   launch_horizontal(g_ctx.d_in,  g_ctx.d_tmp, ncols, nrows, gaussderiv_kernel, g_ctx.stream);
//   launch_vertical  (g_ctx.d_tmp, g_ctx.d_out, ncols, nrows, gauss_kernel,      g_ctx.stream);
//   KLT_CUDA_ASSERT(cudaMemcpyAsync(gradx->data, g_ctx.d_out, bytes, cudaMemcpyDeviceToHost, g_ctx.stream));

//   // grady: (gauss then gaussderiv), reusing the same uploaded d_in
//   launch_horizontal(g_ctx.d_in,  g_ctx.d_tmp, ncols, nrows, gauss_kernel,      g_ctx.stream);
//   launch_vertical  (g_ctx.d_tmp, g_ctx.d_out, ncols, nrows, gaussderiv_kernel, g_ctx.stream);
//   KLT_CUDA_ASSERT(cudaMemcpyAsync(grady->data, g_ctx.d_out, bytes, cudaMemcpyDeviceToHost, g_ctx.stream));

//   KLT_CUDA_ASSERT(cudaStreamSynchronize(g_ctx.stream));
// }

// void _KLTComputeSmoothedImage(
//   _KLT_FloatImage img,
//   float sigma,
//   _KLT_FloatImage smooth)
// {
//   assert(smooth->ncols >= img->ncols);
//   assert(smooth->nrows >= img->nrows);

//   if (fabsf(sigma - sigma_last) > 0.05f)
//     _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

//   _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
// }





