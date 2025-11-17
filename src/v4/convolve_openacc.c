/*********************************************************************
 * convolve_openacc.c
 * GCC-Compatible High-Performance OpenACC Implementation
 * 
 * KEY FIXES:
 * 1. Separate gang/vector directives - no combined gang vector
 * 2. Use parallel region with explicit loop directives
 * 3. Optimized border handling for better memory access
 *********************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <openacc.h>

#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"

#define MAX_KERNEL_WIDTH    71

typedef struct {
    int width;
    float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

/* Kernels - persistently resident on device */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0f;
static int kernels_on_device = 0;

void _KLTToFloatImage(KLT_PixelType *img, int ncols, int nrows, _KLT_FloatImage floatimg) {
    int total_pixels = ncols * nrows;
    floatimg->ncols = ncols;
    floatimg->nrows = nrows;
    
    // Async copy to device
    #pragma acc parallel loop copyin(img[0:total_pixels]) copyout(floatimg->data[0:total_pixels]) async(1)
    for (int i = 0; i < total_pixels; i++) {
        floatimg->data[i] = (float) img[i];
    }
    #pragma acc wait(1)
}

/* Host-only kernel computation */
static void _computeKernels(float sigma, ConvolutionKernel *gauss, ConvolutionKernel *gaussderiv) {
    const float factor = 0.01f;
    int i;
    const int hw = MAX_KERNEL_WIDTH / 2;

    // Compute unnormalized kernels
    for (i = -hw; i <= hw; i++) {
        gauss->data[i+hw] = expf(-i*i / (2.0f*sigma*sigma));
        gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }

    // Truncate tails
    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabsf(gauss->data[i+hw]/1.0f) < factor; i++, gauss->width-=2);
    
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabsf(gaussderiv->data[i+hw]/(sigma*expf(-0.5f))) < factor; i++, gaussderiv->width-=2);

    // Shift to start at index 0
    int offset = (MAX_KERNEL_WIDTH - gauss->width) / 2;
    for (i = 0; i < gauss->width; i++)
        gauss->data[i] = gauss->data[i + offset];
    
    offset = (MAX_KERNEL_WIDTH - gaussderiv->width) / 2;
    for (i = 0; i < gaussderiv->width; i++)
        gaussderiv->data[i] = gaussderiv->data[i + offset];

    // Normalize
    float den = 0.0f;
    for (i = 0; i < gauss->width; i++) den += gauss->data[i];
    for (i = 0; i < gauss->width; i++) gauss->data[i] /= den;

    den = 0.0f;
    int hw_deriv = gaussderiv->width/2;
    for (i = -hw_deriv; i <= hw_deriv; i++) 
        den -= i * gaussderiv->data[i+hw_deriv];
    for (i = -hw_deriv; i <= hw_deriv; i++) 
        gaussderiv->data[i+hw_deriv] /= den;

    sigma_last = sigma;

    // Copy kernels to device ONCE
    if (!kernels_on_device) {
        #pragma acc enter data copyin(gauss_kernel, gaussderiv_kernel)
        kernels_on_device = 1;
    } else {
        #pragma acc update device(gauss_kernel, gaussderiv_kernel)
    }
}

void _KLTGetKernelWidths(float sigma, int *gauss_width, int *gaussderiv_width) {
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
    *gauss_width = gauss_kernel.width;
    *gaussderiv_width = gaussderiv_kernel.width;
}

/* HORIZONTAL CONVOLUTION - FIXED GCC COMPATIBILITY */
static void _convolveImageHoriz(_KLT_FloatImage imgin, ConvolutionKernel kernel, _KLT_FloatImage imgout) {
    int ncols = imgin->ncols, nrows = imgin->nrows;
    int kw = kernel.width, radius = kw/2;
    int i, j;
    int img_size = ncols*nrows;
    float *in = imgin->data, *out = imgout->data;

    assert(imgin != imgout);
    assert(imgout->ncols >= ncols && imgout->nrows >= nrows);

    // FIX: Remove 'vector' from parallel directive, use only on inner loop
    #pragma acc parallel present(in[0:img_size], out[0:img_size], kernel.data[0:kw]) async(1)
    {
        #pragma acc loop gang
        for (j = 0; j < nrows; j++) {
            int base = j * ncols;
            
            // Vectorize across columns for coalesced access
            #pragma acc loop vector independent
            for (i = 0; i < ncols; i++) {
                if (i < radius || i >= ncols - radius) {
                    out[base + i] = 0.0f;
                } else {
                    float sum = 0.0f;
                    int start = base + i - radius;
                    #pragma acc loop seq
                    for (int t = 0; t < kw; t++) {
                        sum += in[start + t] * kernel.data[kw - 1 - t];
                    }
                    out[base + i] = sum;
                }
            }
        }
    }
    #pragma acc wait(1)
}

/* VERTICAL CONVOLUTION - FIXED GCC COMPATIBILITY */
static void _convolveImageVert(_KLT_FloatImage imgin, ConvolutionKernel kernel, _KLT_FloatImage imgout) {
    int ncols = imgin->ncols, nrows = imgin->nrows;
    int kw = kernel.width, radius = kw/2;
    int i, j;
    int img_size = ncols*nrows;
    float *in = imgin->data, *out = imgout->data;

    assert(imgin != imgout);
    assert(imgout->ncols >= ncols && imgout->nrows >= nrows);

    #pragma acc parallel present(in[0:img_size], out[0:img_size], kernel.data[0:kw]) async(2)
    {
        // Process main region (non-borders)
        #pragma acc loop gang
        for (j = radius; j < nrows - radius; j++) {
            #pragma acc loop vector independent
            for (i = 0; i < ncols; i++) {
                float sum = 0.0f;
                #pragma acc loop seq
                for (int t = 0; t < kw; t++) {
                    sum += in[(j - radius + t) * ncols + i] * kernel.data[kw - 1 - t];
                }
                out[j * ncols + i] = sum;
            }
        }

        // Top border - separate loop for simplicity
        #pragma acc loop gang
        for (j = 0; j < radius; j++) {
            #pragma acc loop vector independent
            for (i = 0; i < ncols; i++) {
                out[j * ncols + i] = 0.0f;
            }
        }

        // Bottom border
        #pragma acc loop gang
        for (j = nrows - radius; j < nrows; j++) {
            #pragma acc loop vector independent
            for (i = 0; i < ncols; i++) {
                out[j * ncols + i] = 0.0f;
            }
        }
    }
    #pragma acc wait(2)
}

/* SEPARATE CONVOLUTION - Device-only temp buffer */
static void _convolveSeparate(_KLT_FloatImage imgin, ConvolutionKernel horiz_kernel, ConvolutionKernel vert_kernel, _KLT_FloatImage imgout) {
    int ncols = imgin->ncols, nrows = imgin->nrows;
    int img_size = ncols * nrows;
    
    // Allocate device memory directly
    float *tmp_data = (float*)acc_malloc(img_size * sizeof(float));
    
    _KLT_FloatImageRec tmpimg_rec;
    tmpimg_rec.data = tmp_data;
    tmpimg_rec.ncols = ncols;
    tmpimg_rec.nrows = nrows;
    _KLT_FloatImage tmpimg = &tmpimg_rec;
    
    #pragma acc enter data create(tmpimg->data[0:img_size])
    
    _convolveImageHoriz(imgin, horiz_kernel, tmpimg);
    _convolveImageVert(tmpimg, vert_kernel, imgout);
    
    #pragma acc exit data delete(tmpimg->data[0:img_size])
    acc_free(tmp_data);
}

/* TOP-LEVEL FUNCTIONS */
void _KLTComputeGradients(_KLT_FloatImage img, float sigma, _KLT_FloatImage gradx, _KLT_FloatImage grady) {
    assert(gradx->ncols >= img->ncols && gradx->nrows >= img->nrows);
    assert(grady->ncols >= img->ncols && grady->nrows >= img->nrows);

    if (fabsf(sigma - sigma_last) > 0.05f)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

    int img_size = img->ncols * img->nrows;
    
    #pragma acc data copyin(img->data[0:img_size]) \
                       copyout(gradx->data[0:img_size], grady->data[0:img_size])
    {
        _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
        _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);
    }
}

void _KLTComputeSmoothedImage(_KLT_FloatImage img, float sigma, _KLT_FloatImage smooth) {
    assert(smooth->ncols >= img->ncols && smooth->nrows >= img->nrows);

    if (fabsf(sigma - sigma_last) > 0.05f)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

    int img_size = img->ncols * img->nrows;
    
    #pragma acc data copyin(img->data[0:img_size]) copyout(smooth->data[0:img_size])
    {
        _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
    }
}

/* Cleanup */
void _KLTConvolveCleanup(void) {
    if (kernels_on_device) {
        #pragma acc exit data delete(gauss_kernel, gaussderiv_kernel)
        kernels_on_device = 0;
    }
}