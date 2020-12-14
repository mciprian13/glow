/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "arm_cm_defs.h"

void libjit_arm_cm_conv2d_1x1_i8xi8(      int8_t *outPtr,
                                    const int8_t *inpPtr,
                                    const int8_t *fltPtr,
                                    const int32_t *biasPtr,

                                    const int32_t inpRows,
                                    const int32_t fltRows,
                                    const int32_t numCols,

                                    const int32_t outOffset,
                                    const int32_t inpOffset,
                                    const int32_t fltOffset,

                                    const int32_t outPre,
                                    const int32_t outPost,
                                    const int32_t outScale) {

  arm_cm_mat_mul_rhst_i8xi8(inpPtr,    // const int8_t *lhsPtr
                            fltPtr,    // const int8_t *rhsPtr
                            biasPtr,   // const int32_t *biasPtr
                            outPtr,    //       int8_t *outPtr
                            inpRows,   // const int32_t lhsRows
                            fltRows,   // const int32_t rhsRows
                            numCols,   // const int32_t numCols
                            inpOffset, // const int32_t lhsOffset
                            fltOffset, // const int32_t rhsOffset
                            outOffset, // const int32_t outOffset
                            outPre,    // const int32_t outPre
                            outPost,   // const int32_t outPost
                            outScale,  // const int32_t outScale
                            0,         // const int32_t *outMultipliers
                            0          // const int32_t *outShifters
                            );
}


void libjit_arm_cm_conv2d_dw_i8xi8(      int8_t *outPtr,
                                   const int8_t *inpPtr,
                                   const int8_t *fltPtr,
                                   const int32_t *biasPtr,

                                   const int32_t outH,
                                   const int32_t outW,

                                   const int32_t inpN,
                                   const int32_t inpH,
                                   const int32_t inpW,
                                   const int32_t inpC,

                                   const int32_t kernelH,
                                   const int32_t kernelW,

                                   const int32_t strideH,
                                   const int32_t strideW,

                                   const int32_t padT,
                                   const int32_t padL,

                                   const int32_t outOffset,
                                   const int32_t inpOffset,
                                   const int32_t fltOffset,

                                   const int32_t outPre,
                                   const int32_t outPost,
                                   const int32_t outScale
                                   ) {

  // For each input in the batch.
  for (int32_t n = 0; n < inpN; n++) {

    // For each output height.
    int32_t i_h_min = -padT;
    for (int32_t o_h = 0; o_h < outH; o_h++, i_h_min += strideH) {

      int32_t f_h_min = MAX(      0,      - i_h_min);
      int32_t f_h_max = MIN(kernelH, inpH - i_h_min);
      int32_t f_h_len = f_h_max - f_h_min;

      const int8_t *fltPtrH = fltPtr + f_h_min * kernelW;
      const int8_t *inpPtrH = inpPtr + (i_h_min + f_h_min) * inpW * inpC;

      // For each output width.
      int32_t i_w_min = -padL;
      for (int32_t o_w = 0; o_w < outW; o_w++, i_w_min += strideW) {

        int32_t f_w_min = MAX(      0,      - i_w_min);
        int32_t f_w_max = MIN(kernelW, inpW - i_w_min);
        int32_t f_w_len = f_w_max - f_w_min;

        const int8_t *fltPtr = fltPtrH + f_w_min;
        const int8_t *inpPtr = inpPtrH + (i_w_min + f_w_min) * inpC;

        // Backup pointers.
        const int8_t *fltPtrSave = fltPtr;
        const int8_t *inpPtrSave = inpPtr;

        // For each output channel.
        for (int32_t o_c = 0; o_c < inpC; o_c++) {

          // Initialize sum.
          int32_t sum = biasPtr[o_c];

          // For each filter height.
          for (int32_t f_h = 0; f_h < f_h_len; f_h++) {

            // For each filter width.
            for (int32_t f_w = 0; f_w < f_w_len; f_w++) {

              // Accumulate along the filter height/width plane.
              sum += (*fltPtr - fltOffset) * (*inpPtr - inpOffset);

              // Advance pointers for next filter width.
              fltPtr++;
              inpPtr += inpC;
            }

            // Advance pointers for next filter height.
            fltPtr = fltPtr - f_w_len + kernelW;
            inpPtr = inpPtr - f_w_len * inpC + inpW * inpC;
          }

          // Write output.
          *outPtr++ = __SSAT8(arm_cm_scale_i32i8(sum, outPre, outPost, outScale, outOffset));

          // Advance pointers for next output channel.
          fltPtr = fltPtrSave + (o_c + 1) * kernelH * kernelW;
          inpPtr = inpPtrSave + (o_c + 1) * 1;
        }
      }
    }

    // Advance input pointer for next batch.
    inpPtr += inpH * inpW * inpC;
  }
}
