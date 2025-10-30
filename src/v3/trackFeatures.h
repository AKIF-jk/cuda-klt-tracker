/*********************************************************************
 * trackFeatures.h — GPU/CPU KLT Feature Tracking Interface
 *********************************************************************/

#ifndef _TRACK_FEATURES_H_
#define _TRACK_FEATURES_H_

#include "base.h"
#include "error.h"
#include "klt.h"
#include "klt_util.h"
#include "pyramid.h"
#include "convolve.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
   Main Tracking Function (implemented in trackFeatures.c/.cu)
   ============================================================ */

/**
 * Tracks features from img1 to img2 using gradient information.
 */
void KLTTrackFeatures(
  _KLT_FloatImage img1,
  _KLT_FloatImage img2,
  _KLT_FloatImage gradx1,
  _KLT_FloatImage grady1,
  _KLT_FloatImage gradx2,
  _KLT_FloatImage grady2,
  KLT_FeatureList fl,
  KLT_TrackingContext tc);

/* ============================================================
   GPU Accelerated Internal Helpers
   ============================================================ */

/**
 * GPU kernel for computing intensity difference between patches.
 */
void _computeIntensityDifference_cuda(
  _KLT_FloatImage img1,
  _KLT_FloatImage img2,
  float x1, float y1,
  float x2, float y2,
  int width, int height,
  float *diff);

/**
 * GPU kernel for computing combined gradient sums.
 */
void _computeGradientSum_cuda(
  _KLT_FloatImage gx1,
  _KLT_FloatImage gy1,
  _KLT_FloatImage gx2,
  _KLT_FloatImage gy2,
  float x1, float y1,
  float x2, float y2,
  int width, int height,
  float *gradx, float *grady);

/* ============================================================
   CPU fallback helpers (for reference/testing)
   ============================================================ */
void _computeIntensityDifference_cpu(
  _KLT_FloatImage img1,
  _KLT_FloatImage img2,
  float x1, float y1,
  float x2, float y2,
  int width, int height,
  float *diff);

/* ============================================================
   Temporary floating window buffer type
   (used internally for patch data)
   ============================================================ */
typedef float* _FloatWindow;

#ifdef __cplusplus
}
#endif

#endif /* _TRACK_FEATURES_H_ */
