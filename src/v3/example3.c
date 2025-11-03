/**********************************************************************
Finds the 150 best features in an image and tracks them through the 
next two images.  The sequential mode is set in order to speed
processing.  The features are stored in a feature table, which is then
saved to a text file; each feature list is also written to a PPM file.
**********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include "pnmio.h"
#include "klt.h"
#include <sys/time.h>
#include "klt_cuda_mem.h"


/* #define REPLACE */

#ifdef WIN32
int RunExample3()
#else
int main()
#endif
{
  unsigned char *img1, *img2;
  char fnamein[100], fnameout[100];
  KLT_TrackingContext tc;
  KLT_FeatureList fl;
  KLT_FeatureTable ft;
  int nFeatures = 5000, nFrames = 100;
  int ncols, nrows;
  int i;

  tc = KLTCreateTrackingContext();
  fl = KLTCreateFeatureList(nFeatures);
  ft = KLTCreateFeatureTable(nFrames, nFeatures);
  tc->sequentialMode = TRUE;
  tc->writeInternalImages = FALSE;
  tc->affineConsistencyCheck = -1;  /* set this to 2 to turn on affine consistency check */
  struct timeval t0, t1;
  double elapsed;
  double total_timeTracKFeature=0;
 
  img1 = pgmReadFile("../klt/sample/frame_001.pgm", NULL, &ncols, &nrows);
  img2 = (unsigned char *) malloc(ncols*nrows*sizeof(unsigned char));
  
  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
  KLTStoreFeatureList(fl, ft, 0);
  //KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "sample/feat0.pgm");
  for (i = 1 ; i < nFrames ; i++)  {
    
    sprintf(fnamein, "../klt/sample/frame_%03d.pgm", i);
    pgmReadFile(fnamein, img2, &ncols, &nrows);
    gettimeofday(&t0, NULL);   // ---- start timer ----
    KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
    gettimeofday(&t1, NULL);   // ---- end timer ----
    elapsed = (t1.tv_sec - t0.tv_sec) * 1000.0;        // sec → ms
    elapsed += (t1.tv_usec - t0.tv_usec) / 1000.0;     // μs → ms
    total_timeTracKFeature+=elapsed;
    //printf("Iteration %d time: %.3f ms\n", i, elapsed);

#ifdef REPLACE
    KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
#endif
    KLTStoreFeatureList(fl, ft, i);
    sprintf(fnameout, "sample/feat%d.pgm", i);
    //KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, fnameout);

  }
  KLTWriteFeatureTable(ft, "sample/features.txt", "%5.1f");
  KLTWriteFeatureTable(ft, "sample/features.ft", NULL);
  printf("total time by track feature: %.3f ms\n", total_timeTracKFeature);

  KLTFreeFeatureTable(ft);
  KLTFreeFeatureList(fl);
  KLTFreeTrackingContext(tc);
  free(img1);
  free(img2);

  return 0;
}

