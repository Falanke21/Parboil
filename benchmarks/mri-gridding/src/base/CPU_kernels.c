/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <immintrin.h>

#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#define PI 3.14159265359

float kernel_value_CPU(float v){

  float rValue = 0;

  const float z = v*v;

  // polynomials taken from http://ccrma.stanford.edu/CCRMA/Courses/422/projects/kbd/kbdwindow.cpp
  float num = (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z*
  (z* 0.210580722890567e-22f  + 0.380715242345326e-19f ) +
   0.479440257548300e-16f) + 0.435125971262668e-13f ) +
   0.300931127112960e-10f) + 0.160224679395361e-7f  ) +
   0.654858370096785e-5f)  + 0.202591084143397e-2f  ) +
   0.463076284721000e0f)   + 0.754337328948189e2f   ) +
   0.830792541809429e4f)   + 0.571661130563785e6f   ) +
   0.216415572361227e8f)   + 0.356644482244025e9f   ) +
   0.144048298227235e10f);

  float den = (z*(z*(z-0.307646912682801e4f)+0.347626332405882e7f)-0.144048298227235e10f);

  rValue = -num/den;

  return rValue;
}

void calculateLUT(float beta, float width, float** LUT, unsigned int* sizeLUT){
  float v;
  float cutoff2 = (width*width)/4.0;

  unsigned int size;

  if(width > 0){
    // compute size of LUT based on kernel width
    size = (unsigned int)(10000*width);

    // allocate memory
    (*LUT) = (float*) malloc (size*sizeof(float));

    unsigned int k;
    for(k=0; k<size; ++k){
      // compute value to evaluate kernel at
      // v in the range 0:(_width/2)^2
      v = (((float)k)/((float)size))*cutoff2;

      // compute kernel value and store
      (*LUT)[k] = kernel_value_CPU(beta*sqrt(1.0-(v/cutoff2)));
    }
    (*sizeLUT) = size;
  }
}

float kernel_value_LUT(float v, float* LUT, int sizeLUT, float _1overCutoff2)
{
  unsigned int k0;
  float v0;

  v *= (float)sizeLUT;
  k0=(unsigned int)(v*_1overCutoff2);
  v0 = ((float)k0)/_1overCutoff2;
  return  LUT[k0] + ((v-v0)*(LUT[k0+1]-LUT[k0])/_1overCutoff2);
}

int gridding_Gold(unsigned int n, parameters params, ReconstructionSample* __restrict__ sample, float* __restrict__ LUT, unsigned int sizeLUT, cmplx* __restrict__ gridData, float* __restrict__ sampleDensity){

  unsigned int NxL, NxH;
  unsigned int NyL, NyH;
  unsigned int NzL, NzH;

  int nx;
  int ny;
  int nz;

  float w;
  unsigned int idx;
  unsigned int idx0;

  unsigned int idxZ;
  unsigned int idxY;

  float Dx2[100];
  float Dy2[100];
  float Dz2[100];
  float *dx2=NULL;
  float *dy2=NULL;
  float *dz2=NULL;

  float dy2dz2;
  float v;

  unsigned int size_x = params.gridSize[0];
  unsigned int size_y = params.gridSize[1];
  unsigned int size_z = params.gridSize[2];

  float cutoff = ((float)(params.kernelWidth))/2.0; // cutoff radius
  float cutoff2 = cutoff*cutoff;                    // square of cutoff radius
  float _1overCutoff2 = 1/cutoff2;                  // 1 over square of cutoff radius

  float beta = PI * sqrt(4*params.kernelWidth*params.kernelWidth/(params.oversample*params.oversample) * (params.oversample-.5)*(params.oversample-.5)-.8);

  int i;
  for (i=0; i < n; i++){
    ReconstructionSample pt = sample[i];

    float kx = pt.kX;
    float ky = pt.kY;
    float kz = pt.kZ;

    NxL = max((kx - cutoff), 0.0);
    NxH = min((kx + cutoff), size_x-1.0);

    NyL = max((ky - cutoff), 0.0);
    NyH = min((ky + cutoff), size_y-1.0);

    NzL = max((kz - cutoff), 0.0);
    NzH = min((kz + cutoff), size_z-1.0);

    if((pt.real != 0.0 || pt.imag != 0.0) && pt.sdc!=0.0)
    {
      for(dz2 = Dz2, nz=NzL; nz<=NzH; ++nz, ++dz2)
      {
        *dz2 = ((kz-nz)*(kz-nz));
      }
      for(dx2=Dx2,nx=NxL; nx<=NxH; ++nx,++dx2)
      {
        *dx2 = ((kx-nx)*(kx-nx));
      }
      for(dy2=Dy2, ny=NyL; ny<=NyH; ++ny,++dy2)
      {
        *dy2 = ((ky-ny)*(ky-ny));
      }

      idxZ = (NzL-1)*size_x*size_y;
      for(dz2=Dz2, nz=NzL; nz<=NzH; ++nz, ++dz2)
      {
        /* linear offset into 3-D matrix to get to zposition */
        idxZ += size_x*size_y;

        idxY = (NyL-1)*size_x;

        /* loop over x indexes, but only if curent distance is close enough (distance will increase by adding x&y distance) */
        if((*dz2)<cutoff2)
        {
          for(dy2=Dy2, ny=NyL; ny<=NyH; ++ny, ++dy2)
          {
            /* linear offset IN ADDITION to idxZ to get to Y position */
            idxY += size_x;

            dy2dz2=(*dz2)+(*dy2);

            idx0 = idxY + idxZ;

            /* loop over y indexes, but only if curent distance is close enough (distance will increase by adding y distance) */
            if(dy2dz2<cutoff2)
            {
              int len = NxH - NxL + 1;

                // Start rt-check
                int stride = 4;
                int upper_bound = len / stride * stride;
                int i = 0;
                __m128 vec_NxL = _mm_set1_ps(NxL);
                __m128 vec_dy2dz2 = _mm_set1_ps(dy2dz2);
                __m128 vec_dx2 = _mm_set1_ps(*dx2);
                for (; i < upper_bound; i += stride) {
                  __m128 vec_nx = vec_NxL + (__m128){i, i+1, i+2, i+3};
                  __m128 vec_v = vec_dy2dz2 + vec_dx2;
                  if (vec_v[0]<cutoff2 && vec_v[1]<cutoff2 && vec_v[2]<cutoff2 && vec_v[3]<cutoff2) 
                  {
                    __m128 vec_idx = vec_nx + (float)idx0;
                    __m128 vec_val = beta * _mm_sqrt_ps(1.0 - (vec_v * _1overCutoff2));
                    __m128 vec_z = vec_val * vec_val;
                    __m128 vec_num = (vec_z * (vec_z * (vec_z * (vec_z * (vec_z * (vec_z * (vec_z * (vec_z * (vec_z * (vec_z * (vec_z * (vec_z * (vec_z *
                                                                                                 (vec_z *
                                                                                                  0.210580722890567e-22f +
                                                                                                  0.380715242345326e-19f) +
                                                                                                 0.479440257548300e-16f) +
                                                                                            0.435125971262668e-13f) +
                                                                                       0.300931127112960e-10f) +
                                                                                  0.160224679395361e-7f) +
                                                                             0.654858370096785e-5f) +
                                                                        0.202591084143397e-2f) +
                                                                   0.463076284721000e0f) + 0.754337328948189e2f) +
                                                         0.830792541809429e4f) + 0.571661130563785e6f) +
                                               0.216415572361227e8f) + 0.356644482244025e9f) +
                                     0.144048298227235e10f);
                    __m128 vec_den = (vec_z * (vec_z * (vec_z - 0.307646912682801e4f) + 0.347626332405882e7f) -
                                     0.144048298227235e10f);
                    __m128 vec_rValue = (0 - vec_num) / vec_den;
                    __m128 vec_w = vec_rValue * pt.sdc;
                    __m128 vec_w_real_product = vec_w * pt.real;
                    __m128 vec_w_imag_product = vec_w * pt.imag;
                    for (int rtcheck_i = 0; rtcheck_i < stride; rtcheck_i++) {
                      idx = (unsigned int)vec_idx[rtcheck_i];
                      gridData[idx].real += vec_w_real_product[rtcheck_i];
                      gridData[idx].imag += vec_w_imag_product[rtcheck_i];
                      sampleDensity[idx] += 1.0;
                    }
                  }
                  else if (!(vec_v[0]<cutoff2) && !(vec_v[1]<cutoff2) && !(vec_v[2]<cutoff2) && !(vec_v[3]<cutoff2))
                  {}
                  else
                  {
                    // Unroll
                    for (int unroll_i = 0; unroll_i < stride; unroll_i++) {
                      if (vec_v[unroll_i] < cutoff2) {
                          idx = vec_nx[unroll_i] + idx0;
                          float val = beta*sqrt(1.0-(vec_v[unroll_i]*_1overCutoff2));
                          float rValue = 0;

                          const float z = val * val;

                          // polynomials taken from http://ccrma.stanford.edu/CCRMA/Courses/422/projects/kbd/kbdwindow.cpp
                          float num = (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z *
                                                                                                    (z *
                                                                                                    0.210580722890567e-22f +
                                                                                                    0.380715242345326e-19f) +
                                                                                                    0.479440257548300e-16f) +
                                                                                              0.435125971262668e-13f) +
                                                                                          0.300931127112960e-10f) +
                                                                                    0.160224679395361e-7f) +
                                                                                0.654858370096785e-5f) +
                                                                          0.202591084143397e-2f) +
                                                                      0.463076284721000e0f) + 0.754337328948189e2f) +
                                                            0.830792541809429e4f) + 0.571661130563785e6f) +
                                                  0.216415572361227e8f) + 0.356644482244025e9f) +
                                        0.144048298227235e10f);

                          float den = (z * (z * (z - 0.307646912682801e4f) + 0.347626332405882e7f) -
                                        0.144048298227235e10f);

                          rValue = -num / den;
                          w = rValue * pt.sdc;

                          /* grid data */
                          gridData[idx].real += (w * pt.real);
                          gridData[idx].imag += (w * pt.imag);

                          /* estimate sample density */
                          sampleDensity[idx] += 1.0;
                      }
                    }
                  }
                }
                // Handle remainder
                for (; i < len; ++i) {
                    nx = NxL + i;
                    /* value to evaluate kernel at */
                    v = dy2dz2 + (*dx2);
                    if (v < cutoff2) {
                        idx = nx + idx0;
                        float val = beta*sqrt(1.0-(v*_1overCutoff2));
                        float rValue = 0;
                        const float z = val * val;
                        // polynomials taken from http://ccrma.stanford.edu/CCRMA/Courses/422/projects/kbd/kbdwindow.cpp
                        float num = (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z *
                                                                                                 (z *
                                                                                                  0.210580722890567e-22f +
                                                                                                  0.380715242345326e-19f) +
                                                                                                 0.479440257548300e-16f) +
                                                                                            0.435125971262668e-13f) +
                                                                                       0.300931127112960e-10f) +
                                                                                  0.160224679395361e-7f) +
                                                                             0.654858370096785e-5f) +
                                                                        0.202591084143397e-2f) +
                                                                   0.463076284721000e0f) + 0.754337328948189e2f) +
                                                         0.830792541809429e4f) + 0.571661130563785e6f) +
                                               0.216415572361227e8f) + 0.356644482244025e9f) +
                                     0.144048298227235e10f);
                        float den = (z * (z * (z - 0.307646912682801e4f) + 0.347626332405882e7f) -
                                     0.144048298227235e10f);
                        rValue = -num / den;
                        w = rValue * pt.sdc;
                        /* grid data */
                        gridData[idx].real += (w * pt.real);
                        gridData[idx].imag += (w * pt.imag);
                        /* estimate sample density */
                        sampleDensity[idx] += 1.0;
                    }
                }
                // End rt-check

                // Scalar code
                // for (int i = 0; i < len; ++i) {
                //     nx = NxL + i;
                //     /* value to evaluate kernel at */
                //     v = dy2dz2 + (*dx2);
                //     if (v < cutoff2) {
                //         idx = nx + idx0;
                //         float val = beta*sqrt(1.0-(v*_1overCutoff2));
                //         float rValue = 0;
                //         const float z = val * val;
                //         // polynomials taken from http://ccrma.stanford.edu/CCRMA/Courses/422/projects/kbd/kbdwindow.cpp
                //         float num = (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z *
                //                                                                                  (z *
                //                                                                                   0.210580722890567e-22f +
                //                                                                                   0.380715242345326e-19f) +
                //                                                                                  0.479440257548300e-16f) +
                //                                                                             0.435125971262668e-13f) +
                //                                                                        0.300931127112960e-10f) +
                //                                                                   0.160224679395361e-7f) +
                //                                                              0.654858370096785e-5f) +
                //                                                         0.202591084143397e-2f) +
                //                                                    0.463076284721000e0f) + 0.754337328948189e2f) +
                //                                          0.830792541809429e4f) + 0.571661130563785e6f) +
                //                                0.216415572361227e8f) + 0.356644482244025e9f) +
                //                      0.144048298227235e10f);
                //         float den = (z * (z * (z - 0.307646912682801e4f) + 0.347626332405882e7f) -
                //                      0.144048298227235e10f);
                //         rValue = -num / den;
                //         w = rValue * pt.sdc;
                //         /* grid data */
                //         gridData[idx].real += (w * pt.real);
                //         gridData[idx].imag += (w * pt.imag);
                //         /* estimate sample density */
                //         sampleDensity[idx] += 1.0;
                //     }
                // }
            }
          }
        }
      }
    }
  }
}
