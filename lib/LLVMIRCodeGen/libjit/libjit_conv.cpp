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
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "libjit_defs.h"

namespace {
// Initialize the convolution output frame for slice \p N with the bias \p
// biasW.
void libjit_conv_init_output_with_bias(dim_t N, float *outW, const float *biasW,
                                       const dim_t *outWdims,
                                       const dim_t *biasWdims) {
  // For each (x,y) step in the output tensor:
  for (dim_t ax = 0; ax < outWdims[1]; ax++) {
    for (dim_t ay = 0; ay < outWdims[2]; ay++) {
      // For each output channel:
      for (dim_t d = 0; d < outWdims[3]; d++) {
        // Store the results to the output buffer.
        float bias = biasW[d];
        auto outIdx = libjit_getXYZW(outWdims, N, ax, ay, d);
        outW[outIdx] = bias;
      } // For each depth in the output.
    }   // For each Y in the output.
  }     // For each X in the output.
}

#if 0
/// Generic template for quantized conv2d. The template allows choosing
/// element type and bias type.
template <typename ElemTy, typename BiasElemTy>
void libjit_quantized_conv2d_generic(
    ElemTy *outW, const ElemTy *inW, const ElemTy *filterW,
    const BiasElemTy *biasW, const dim_t *outWdims, const dim_t *inWdims,
    const dim_t *filterWdims, const dim_t *biasWdims, const dim_t *kernelSizes,
    const dim_t *strides, const dim_t *pads, dim_t group, int32_t outOffset,
    int32_t inOffset, int32_t filterOffset, int32_t biasOffset, int32_t biasPre,
    int32_t biasPost, int32_t biasScale, int32_t outPre, int32_t outPost,
    int32_t outScale, unsigned depthUnroll, dim_t dilation) {
  dim_t inChannels = inWdims[3];
  dim_t outChannels = outWdims[3];
  dim_t inCperG = inChannels / group;
  dim_t outCperG = outChannels / group;
  dim_t pad_t = pads[0];
  dim_t pad_l = pads[1];
  dim_t stride_h = strides[0];
  size_t stride_w = strides[1];
  size_t kernel_h = kernelSizes[0];
  size_t kernel_w = kernelSizes[1];
  // For each input in the batch:
  for (size_t n = 0; n < inWdims[0]; n++) {
    // For each group of input channels:
    for (size_t g = 0; g < group; g++) {

      // For each output channel in the group. Process 'depthUnroll' output
      // layers together.
      for (size_t d = g * outCperG; d < (g + 1) * outCperG; d += depthUnroll) {
        // For each convolution 'jump' in the input tensor:
        ssize_t x = -(ssize_t)pad_t;
        for (size_t ax = 0; ax < outWdims[1]; x += stride_h, ax++) {
          ssize_t y = -(ssize_t)pad_l;
          for (size_t ay = 0; ay < outWdims[2]; y += stride_w, ay++) {
            int32_t sum[depthUnroll];

            for (unsigned i = 0; i < depthUnroll; i++) {
              // Scale the bias to match the scale of the matrix multiplication.
              sum[i] = libjit_scale_i32i8((int32_t)biasW[d + i] - biasOffset,
                                          biasPre, biasPost, biasScale, 0);
            }

            // For each element in the convolution-filter:
            for (size_t fx = 0; fx < kernel_h; fx++) {
              for (size_t fy = 0; fy < kernel_w; fy++) {
                ssize_t ox = x + fx * dilation;
                ssize_t oy = y + fy * dilation;

                // Ignore index access below zero (this is due to padding).
                if (ox < 0 || oy < 0 || ox >= (ssize_t)inWdims[1] ||
                    oy >= (ssize_t)inWdims[2]) {
                  continue;
                }

                // Calculate the indices into the Filter and Input buffers.
                size_t inIdx = libjit_getXYZW(inWdims, n, (size_t)ox,
                                              (size_t)oy, g * inCperG);
                size_t filterIdx = libjit_getXYZW(filterWdims, d, fx, fy, 0);
                size_t sliceSize =
                    filterWdims[1] * filterWdims[2] * filterWdims[3];

                // Perform the innermost loop of the convolution using 4 vector
                // registers.
                for (size_t fd = 0; fd < inCperG; fd++) {
                  int32_t in = inW[inIdx + fd] - inOffset;
                  for (unsigned i = 0; i < MIN(4, depthUnroll); i++) {
                    sum[i] += (filterW[filterIdx + (sliceSize * i) + fd] -
                               filterOffset) *
                              in;
                  }
                }

                // And perform the innermost loop again with 4 more registers.
                if (depthUnroll > 4)
                  for (size_t fd = 0; fd < inCperG; fd++) {
                    int32_t in = inW[inIdx + fd] - inOffset;
                    for (unsigned i = 4; i < MIN(8, depthUnroll); i++) {
                      sum[i] += (filterW[filterIdx + (sliceSize * i) + fd] -
                                 filterOffset) *
                                in;
                    }
                  }
              }
            }

            for (unsigned i = 0; i < depthUnroll; i++) {
              // Scale the result back to the expected destination scale.
              int32_t scaledSum = libjit_scale_i32i8(sum[i], outPre, outPost,
                                                     outScale, outOffset);
              outW[libjit_getXYZW(outWdims, n, ax, ay, d + i)] =
                  libjit_clip(scaledSum);
            }
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}

#else

/// Notes:
/// 1. Removing the manual loop-unrolling greatly improves things.
/// 2. Optimizing PAD does not improve anything.
/// 3. 

/// Generic template for quantized conv2d. The template allows choosing
/// element type and bias type.
template <typename ElemTy, typename BiasElemTy>
void libjit_quantized_conv2d_generic(
    ElemTy *outW, const ElemTy *inW, const ElemTy *filterW,
    const BiasElemTy *biasW, const dim_t *outWdims, const dim_t *inWdims,
    const dim_t *filterWdims, const dim_t *biasWdims, const dim_t *kernelSizes,
    const dim_t *strides, const dim_t *pads, dim_t group, int32_t outOffset,
    int32_t inOffset, int32_t filterOffset, int32_t biasOffset, int32_t biasPre,
    int32_t biasPost, int32_t biasScale, int32_t outPre, int32_t outPost,
    int32_t outScale, unsigned depthUnroll, dim_t dilation) {

  dim_t inChannels = inWdims[3];
  dim_t outChannels = outWdims[3];

  dim_t inCperG = inChannels / group;
  dim_t outCperG = outChannels / group;

  dim_t padT = pads[0];
  dim_t padL = pads[1];

  size_t strideH = strides[0];
  size_t strideW = strides[1];

  size_t kernelH = kernelSizes[0];
  size_t kernelW = kernelSizes[1];

  size_t dilationH = dilation;
  size_t dilationW = dilation;

  // For each input in the batch.
  for (size_t n = 0; n < inWdims[0]; n++) {

    // For each group of input channels.
    for (size_t g = 0; g < group; g++) {

      // For each output channel in the group.
      for (size_t o_c = g * outCperG; o_c < (g + 1) * outCperG; o_c++) {

        // For each output height.
        ssize_t i_h_min = -(ssize_t)padT;
        for (size_t o_h = 0; o_h < outWdims[1]; o_h++, i_h_min += strideH) {

          // For each output width.
          ssize_t i_w_min = -(ssize_t)padL;
          for (size_t o_w = 0; o_w < outWdims[2]; o_w++, i_w_min += strideW) {

            // Scale the bias to match the scale of the matrix multiplication.
            int32_t sum = libjit_scale_i32i8((int32_t)biasW[o_c] - biasOffset, biasPre, biasPost, biasScale, 0);

            // For each filter height.
            for (size_t f_h = 0; f_h < kernelH; f_h++) {

              // Skip due to input H padding.
              ssize_t i_h = i_h_min + f_h * dilationH;
              if (i_h < 0 ||  i_h >= (ssize_t)inWdims[1]) {
                filterW += filterWdims[2] * filterWdims[3];
                continue;
              }

              // For each filter width.
              for (size_t f_w = 0; f_w < kernelW; f_w++) {

                // Skip due to input W padding.
                ssize_t i_w = i_w_min + f_w * dilationW;
                if (i_w < 0 || i_w >= (ssize_t)inWdims[2]) {
                  filterW += filterWdims[3];
                  continue;
                }

                // Compute current indices into filter and input buffers.
                size_t inpIdx = i_h * inWdims[2] * inWdims[3] + i_w * inWdims[3] + g * inCperG;

                // Accumulate along the filter depth.
                for (size_t f_c = 0; f_c < inCperG; f_c++) {
                  sum += (*filterW++ - filterOffset) * (inW[inpIdx + f_c] - inOffset);
                }
              }
            }

            // Compute current index into output buffer.
            size_t outIdx = o_h * outWdims[2] * outWdims[3] + o_w * outWdims[3] + o_c;

            // Scale the result back to the expected destination scale.
            int32_t scaledSum = libjit_scale_i32i8(sum, outPre, outPost, outScale, outOffset);
            outW[outIdx] = libjit_clip(scaledSum);

            // Reset filter pointer for next output height/width.
            filterW -= filterWdims[1] * filterWdims[2] * filterWdims[3];
          } // W
        } // H

        // Advance filter pointer for next output channel.
        filterW += filterWdims[1] * filterWdims[2] * filterWdims[3];

      } // C
    } // G

    // Reset filter pointer for next batch.
    filterW -= filterWdims[0] * filterWdims[1] * filterWdims[2] * filterWdims[3];

    // Advance input/output pointers for next batch.
    inW  += inWdims [1] * inWdims [2] * inWdims [3];
    outW += outWdims[1] * outWdims[2] * outWdims[3];
  } // N
}

#endif

/// Generic template for channelwise quantized conv2d. The template allows
/// choosing the element type and bias type.
template <typename ElemTy, typename BiasElemTy>
void libjit_channelwise_quantized_conv2d_generic(
    ElemTy *outW, const ElemTy *inW, const ElemTy *filterW,
    const BiasElemTy *biasW, const dim_t *outWdims, const dim_t *inWdims,
    const dim_t *filterWdims, const dim_t *biasWdims, const dim_t *kernels,
    const dim_t *strides, const dim_t *pads, dim_t group, dim_t dilation,
    int32_t outOffset, int32_t inOffset, int32_t *filterOffsetsPtr,
    int32_t *biasOffsetsPtr, const int32_t *biasPrePtr,
    const int32_t *biasPostPtr, const int32_t *biasScalePtr,
    const int32_t *outPrePtr, const int32_t *outPostPtr,
    const int32_t *outScalePtr) {

  dim_t inChannels = inWdims[3];
  dim_t outChannels = outWdims[3];
  dim_t inCperG = inChannels / group;
  dim_t outCperG = outChannels / group;
  dim_t pad_t = pads[0];
  dim_t pad_l = pads[1];
  dim_t stride_h = strides[0];
  dim_t stride_w = strides[1];
  dim_t kernel_h = kernels[0];
  dim_t kernel_w = kernels[1];

  // For each input in the batch:
  for (dim_t n = 0; n < inWdims[0]; n++) {
    // For each group of input channels:
    for (dim_t g = 0; g < group; g++) {
      // For each output channel in the group:
      for (dim_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // Get channel wise quantization params.
        int32_t filterOffset = filterOffsetsPtr[d];
        int32_t biasOffset = biasOffsetsPtr[d];
        int32_t biasPre = biasPrePtr[d];
        int32_t biasPost = biasPostPtr[d];
        int32_t biasScale = biasScalePtr[d];
        int32_t outPre = outPrePtr[d];
        int32_t outPost = outPostPtr[d];
        int32_t outScale = outScalePtr[d];

        // For each convolution 'jump' in the input tensor:
        sdim_t x = -(sdim_t)pad_t;
        for (dim_t ax = 0; ax < outWdims[1]; x += stride_h, ax++) {
          sdim_t y = -(sdim_t)pad_l;
          for (dim_t ay = 0; ay < outWdims[2]; y += stride_w, ay++) {

            // Scale the bias to match the scale of the matrix multiplication.
            int32_t sum = libjit_scale_i32i8((int32_t)biasW[d] - biasOffset,
                                             biasPre, biasPost, biasScale, 0);

            // For each element in the convolution-filter:
            for (dim_t fx = 0; fx < kernel_h; fx++) {
              for (dim_t fy = 0; fy < kernel_w; fy++) {
                sdim_t ox = x + fx * dilation;
                sdim_t oy = y + fy * dilation;

                // Ignore access outside the input tensor (due to padding).
                if (ox < 0 || oy < 0 || ox >= (sdim_t)inWdims[1] ||
                    oy >= (sdim_t)inWdims[2]) {
                  continue;
                }

                // Calculate the indices into the Filter and Input buffers.
                dim_t inIdx = libjit_getXYZW(inWdims, n, (dim_t)ox, (dim_t)oy,
                                             g * inCperG);
                dim_t filterIdx = libjit_getXYZW(filterWdims, d, fx, fy, 0);

                // Accumulate along the filter depth.
                for (dim_t fd = 0; fd < inCperG; fd++) {
                  sum += (filterW[filterIdx + fd] - filterOffset) *
                         (inW[inIdx + fd] - inOffset);
                }
              }
            }

            // Scale the result back to the expected destination scale.
            int32_t scaledSum =
                libjit_scale_i32i8(sum, outPre, outPost, outScale, outOffset);
            outW[libjit_getXYZW(outWdims, n, ax, ay, d)] =
                libjit_clip(scaledSum);
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}

/// Generic template for channelwise quantized conv3d. The template allows
/// choosing the element type and bias type.
template <typename ElemTy, typename BiasElemTy>
void libjit_channelwise_quantized_conv3d_generic(
    ElemTy *outW, const ElemTy *inW, const ElemTy *filterW,
    const BiasElemTy *biasW, const dim_t *outWdims, const dim_t *inWdims,
    const dim_t *filterWdims, const dim_t *biasWdims, const dim_t *kernels,
    const dim_t *strides, const dim_t *pads, dim_t group, dim_t dilation,
    int32_t outOffset, int32_t inOffset, int32_t *filterOffsetsPtr,
    int32_t *biasOffsetsPtr, const int32_t *biasPrePtr,
    const int32_t *biasPostPtr, const int32_t *biasScalePtr,
    const int32_t *outPrePtr, const int32_t *outPostPtr,
    const int32_t *outScalePtr) {

  dim_t inChannels = inWdims[4];
  dim_t outChannels = outWdims[4];
  dim_t inCperG = inChannels / group;
  dim_t outCperG = outChannels / group;

  dim_t pad_near = pads[0];
  dim_t pad_top = pads[2];
  dim_t pad_left = pads[4];

  dim_t stride_t = strides[0];
  dim_t stride_h = strides[1];
  dim_t stride_w = strides[2];

  dim_t kernel_t = kernels[0];
  dim_t kernel_h = kernels[1];
  dim_t kernel_w = kernels[2];

  (void)dilation;

  // For each input in the batch:
  for (dim_t n = 0; n < inWdims[0]; n++) {
    // For each group of input channels:
    for (dim_t g = 0; g < group; g++) {
      // For each output channel in the group:
      for (dim_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // Get channel wise quantization params.
        int32_t filterOffset = filterOffsetsPtr[d];
        int32_t biasOffset = biasOffsetsPtr[d];
        int32_t biasPre = biasPrePtr[d];
        int32_t biasPost = biasPostPtr[d];
        int32_t biasScale = biasScalePtr[d];
        int32_t outPre = outPrePtr[d];
        int32_t outPost = outPostPtr[d];
        int32_t outScale = outScalePtr[d];

        // For each convolution 'jump' in the input tensor:
        sdim_t t = -sdim_t(pad_near);
        for (dim_t at = 0; at < outWdims[1]; t += stride_t, at++) {
          sdim_t x = -sdim_t(pad_top);
          for (dim_t ax = 0; ax < outWdims[2]; x += stride_h, ax++) {
            sdim_t y = -sdim_t(pad_left);
            for (dim_t ay = 0; ay < outWdims[3]; y += stride_w, ay++) {

              // Scale the bias to match the scale of the matrix multiplication.
              int32_t sum = libjit_scale_i32i8((int32_t)biasW[d] - biasOffset,
                                               biasPre, biasPost, biasScale, 0);

              // For each element in the convolution-filter:
              for (dim_t ft = 0; ft < kernel_t; ft++) {
                for (dim_t fx = 0; fx < kernel_h; fx++) {
                  for (dim_t fy = 0; fy < kernel_w; fy++) {
                    sdim_t ot = t + ft;
                    sdim_t ox = x + fx;
                    sdim_t oy = y + fy;

                    // Ignore index access below zero (this is due to
                    // padding).
                    if (ot < 0 || ox < 0 || oy < 0 ||
                        ot >= (sdim_t)inWdims[1] || ox >= (sdim_t)inWdims[2] ||
                        oy >= (sdim_t)inWdims[3]) {
                      continue;
                    }

                    // Calculate the indices into the Filter and Input buffers.
                    dim_t inIdx =
                        libjit_getXYZWQ(inWdims, n, (dim_t)ot, (dim_t)ox,
                                        (dim_t)oy, g * inCperG);
                    dim_t filterIdx =
                        libjit_getXYZWQ(filterWdims, d, ft, fx, fy, 0);

                    // Accumulate along the filter depth.
                    for (dim_t fd = 0; fd < inCperG; fd++) {
                      sum += (filterW[filterIdx + fd] - filterOffset) *
                             (inW[inIdx + fd] - inOffset);
                    }
                  }
                }
              }

              // Scale the result back to the expected destination scale.
              int32_t scaledSum =
                  libjit_scale_i32i8(sum, outPre, outPost, outScale, outOffset);
              outW[libjit_getXYZWQ(outWdims, n, at, ax, ay, d)] =
                  libjit_clip(scaledSum);
            } // W
          }   // H
        }     // T
      }       // C
    }         // G
  }           // N
}
} // namespace

extern "C" {

#if 0

void libjit_conv2d_f(float *outW, const float *inW, const float *filterW,
                     const float *biasW, const dim_t *outWdims,
                     const dim_t *inWdims, const dim_t *filterWdims,
                     const dim_t *biasWdims, const dim_t *kernelSizes,
                     const dim_t *strides, const dim_t *pads, dim_t group,
                     unsigned depthUnroll, dim_t dilation) {
  dim_t inChannels = inWdims[3];
  dim_t outChannels = outWdims[3];
  dim_t inCperG = inChannels / group;
  dim_t outCperG = outChannels / group;

  // The output dims are calculated already from all of the pads,
  // therefore we only need the top and left pads here to control the starting
  // position.
  dim_t pad_t = pads[0];
  dim_t pad_l = pads[1];
  dim_t stride_h = strides[0];
  dim_t stride_w = strides[1];
  dim_t kernel_h = kernelSizes[0];
  dim_t kernel_w = kernelSizes[1];
  // The size of the input-channel tile. High channel count allow for SIMD
  // parallelism but create register pressure. Low channel count reduces the
  // memory pressure and allows things to fit in cache, but require additional
  // compute (horizontal add) to sum the values in the block. This value is a
  // compromise between the two.
  constexpr unsigned cbSize = 512;

  // For each input in the batch:
  for (dim_t n = 0; n < inWdims[0]; n++) {

    // Initialize the output frame for the N'th slice with the bias.
    // Later we will accumulate values into this slice.
    libjit_conv_init_output_with_bias(n, outW, biasW, outWdims, biasWdims);

    // For each group of input channels:
    for (dim_t g = 0; g < group; g++) {
      // Process the body of the loop in tiles of "channel-block".
      for (dim_t cb = 0; cb < inCperG; cb += cbSize) {

        // For each output channel in the group. Process 'depthUnroll' output
        // layers together.
        for (dim_t d = g * outCperG; d < (g + 1) * outCperG; d += depthUnroll) {

          // For each element in the convolution-filter:
          for (dim_t fx = 0; fx < kernel_h; fx++) {
            for (dim_t fy = 0; fy < kernel_w; fy++) {

              // For each convolution 'jump' in the input tensor:
              for (dim_t outx = 0; outx < outWdims[1]; outx++) {
                for (dim_t outy = 0; outy < outWdims[2]; outy++) {

                  // Process 'depthUnroll' output pixels at once. Each scalar
                  // here represents the convolution sum for one (x,y) point in
                  // the output. We process the same pixel for different output
                  // channel (D) values. The compiler should perform scalar
                  // replacement of aggregates and split this tiny array to
                  // registers.
                  float sum[depthUnroll];
                  for (unsigned i = 0; i < depthUnroll; i++) {
                    sum[i] = 0;
                  }

                  // Calculate the specific input x,y that we process in this
                  // iteration.
                  sdim_t inx = (sdim_t)outx * stride_h - pad_t + fx * dilation;
                  sdim_t iny = (sdim_t)outy * stride_w - pad_l + fy * dilation;

                  // Ignore index access below zero (this is due to padding).
                  if (inx < 0 || iny < 0 || inx >= (sdim_t)inWdims[1] ||
                      iny >= (sdim_t)inWdims[2]) {
                    continue;
                  }

                  // Calculate the indices into the Filter and Input buffers.
                  dim_t inIdx = libjit_getXYZW(inWdims, n, (dim_t)inx,
                                               (dim_t)iny, g * inCperG);
                  dim_t filterIdx = libjit_getXYZW(filterWdims, d, fx, fy, 0);
                  dim_t sliceSize =
                      filterWdims[1] * filterWdims[2] * filterWdims[3];

                  // Perform the heart of the convolution, 4 elements at a time
                  // to reduce register pressure.
                  for (dim_t fd = cb, e = MIN(cb + cbSize, inCperG); fd < e;
                       fd++) {
                    float in = inW[inIdx + fd];
                    for (unsigned i = 0; i < MIN(4, depthUnroll); i++) {
                      sum[i] += filterW[filterIdx + (sliceSize * i) + fd] * in;
                    }
                  }

                  // And run the innermost loop again for the second group of
                  // depth slices:
                  if (depthUnroll > 4) {
                    for (dim_t fd = cb, e = MIN(cb + cbSize, inCperG); fd < e;
                         fd++) {
                      float in = inW[inIdx + fd];
                      for (unsigned i = 4; i < MIN(8, depthUnroll); i++) {
                        sum[i] +=
                            filterW[filterIdx + (sliceSize * i) + fd] * in;
                      }
                    }
                  }

                  // Store the results to the output buffer.
                  for (unsigned i = 0; i < depthUnroll; i++) {
                    outW[libjit_getXYZW(outWdims, n, outx, outy, d + i)] +=
                        sum[i];
                  }
                }
              }
            } // For each Y in the filter.
          }   // For each X in the filter.
        }     // For each D (the depth, or the output channel).
      }       // For each block in the input channel.
    }         // For each group in the input channel.
  }           // For each N, the sample in the batch.
}

#else

void libjit_conv2d_f(float *outW, const float *inW, const float *filterW,
                     const float *biasW, const dim_t *outWdims,
                     const dim_t *inWdims, const dim_t *filterWdims,
                     const dim_t *biasWdims, const dim_t *kernelSizes,
                     const dim_t *strides, const dim_t *pads, dim_t group,
                     unsigned depthUnroll, dim_t dilation) {
  dim_t inChannels = inWdims[3];
  dim_t outChannels = outWdims[3];

  dim_t inCperG = inChannels / group;
  dim_t outCperG = outChannels / group;

  dim_t padT = pads[0];
  dim_t padL = pads[1];

  size_t strideH = strides[0];
  size_t strideW = strides[1];

  size_t kernelH = kernelSizes[0];
  size_t kernelW = kernelSizes[1];

  size_t dilationH = dilation;
  size_t dilationW = dilation;

  // For each input in the batch.
  for (size_t n = 0; n < inWdims[0]; n++) {

    // For each group of input channels.
    for (size_t g = 0; g < group; g++) {

      // For each output channel in the group.
      for (size_t o_c = g * outCperG; o_c < (g + 1) * outCperG; o_c++) {

        // For each output height.
        ssize_t i_h_min = -(ssize_t)padT;
        for (size_t o_h = 0; o_h < outWdims[1]; o_h++, i_h_min += strideH) {

          // For each output width.
          ssize_t i_w_min = -(ssize_t)padL;
          for (size_t o_w = 0; o_w < outWdims[2]; o_w++, i_w_min += strideW) {

            // Initialize sum.
            float sum = biasW[o_c];

            // For each filter height.
            for (size_t f_h = 0; f_h < kernelH; f_h++) {

              // Skip due to input H padding.
              ssize_t i_h = i_h_min + f_h * dilationH;
              if (i_h < 0 ||  i_h >= (ssize_t)inWdims[1]) {
                filterW += filterWdims[2] * filterWdims[3];
                continue;
              }

              // For each filter width.
              for (size_t f_w = 0; f_w < kernelW; f_w++) {

                // Skip due to input W padding.
                ssize_t i_w = i_w_min + f_w * dilationW;
                if (i_w < 0 || i_w >= (ssize_t)inWdims[2]) {
                  filterW += filterWdims[3];
                  continue;
                }

                // Compute current indices into filter and input buffers.
                size_t inpIdx = i_h * inWdims[2] * inWdims[3] + i_w * inWdims[3] + g * inCperG;

                // Accumulate along the filter depth.
                for (size_t f_c = 0; f_c < inCperG; f_c++) {
                  sum += (*filterW++) * (inW[inpIdx + f_c]);
                }
              }
            }

            // Compute current index into output buffer.
            size_t outIdx = o_h * outWdims[2] * outWdims[3] + o_w * outWdims[3] + o_c;

            // Write output.
            outW[outIdx] = sum;

            // Reset filter pointer for next output height/width.
            filterW -= filterWdims[1] * filterWdims[2] * filterWdims[3];
          } // W
        } // H

        // Advance filter pointer for next output channel.
        filterW += filterWdims[1] * filterWdims[2] * filterWdims[3];

      } // C
    } // G

    // Reset filter pointer for next batch.
    filterW -= filterWdims[0] * filterWdims[1] * filterWdims[2] * filterWdims[3];

    // Advance input/output pointers for next batch.
    inW  += inWdims [1] * inWdims [2] * inWdims [3];
    outW += outWdims[1] * outWdims[2] * outWdims[3];
  } // N
}

// Kernel limitations:
// - kernel sizes are 1
// - stride sizes are 1
// - dilations are 1
// - paddings are 0
// - group is 1
void libjit_conv2d_1x1_f(float *outW, const float *inW, const float *filterW,
                         const float *biasW, const dim_t *outWdims,
                         const dim_t *inWdims, const dim_t *filterWdims,
                         const dim_t *biasWdims, const dim_t *kernelSizes,
                         const dim_t *strides, const dim_t *pads, dim_t group,
                         unsigned depthUnroll, dim_t dilation) {

  // For each input in the batch.
  for (size_t n = 0; n < inWdims[0]; n++) {

    // For each output channel.
    for (size_t o_c = 0; o_c < outWdims[3]; o_c++) {

      // For each output height/width plane.
      for (size_t o_hw = 0; o_hw < outWdims[1] * outWdims[2]; o_hw++) {

        // Initialize sum.
        float sum = biasW[o_c];

        // Accumulate along the filter depth.
        for (size_t f_c = 0; f_c < inWdims[3]; f_c++) {
          sum += (*filterW++) * (*inW++);
        }

        // Reset filter pointer.
        filterW -= filterWdims[3];

        // Store output.
        *outW = sum;
        outW += outWdims[3];
      }

      // Reset input pointer for next output channel.
      inW -= inWdims [1] * inWdims [2] * inWdims [3];

      // Advance filter pointer for next output channel.
      filterW += filterWdims[3];

      // Advance output pointer for next output channel.
      outW -= outWdims[1] * outWdims[2] * outWdims[3];
      outW++;
    }

    // Reset filter pointer for next batch.
    filterW -= filterWdims[0] * filterWdims[1] * filterWdims[2] * filterWdims[3];

    // Advance input/output pointers for next batch.
    inW  += inWdims [1] * inWdims [2] * inWdims [3];
    outW += outWdims[1] * outWdims[2] * outWdims[3] - outWdims[3];
  } // N
}

#if 0
// V1 Kernel limitations:
// - dilations are 1
// - group = inpC = outC
void libjit_conv2d_dw_f(float *outW, const float *inW, const float *filterW,
                        const float *biasW, const dim_t *outWdims,
                        const dim_t *inWdims, const dim_t *filterWdims,
                        const dim_t *biasWdims, const dim_t *kernelSizes,
                        const dim_t *strides, const dim_t *pads, dim_t group,
                        unsigned depthUnroll, dim_t dilation) {

  dim_t padT = pads[0];
  dim_t padL = pads[1];

  size_t strideH = strides[0];
  size_t strideW = strides[1];

  size_t kernelH = kernelSizes[0];
  size_t kernelW = kernelSizes[1];

  // For each input in the batch.
  for (size_t n = 0; n < inWdims[0]; n++) {

    // For each output channel.
    for (size_t o_c = 0; o_c < outWdims[3]; o_c++) {

      // For each output height.
      ssize_t i_h_min = -(ssize_t)padT;
      for (size_t o_h = 0; o_h < outWdims[1]; o_h++, i_h_min += strideH) {

        // For each output width.
        ssize_t i_w_min = -(ssize_t)padL;
        for (size_t o_w = 0; o_w < outWdims[2]; o_w++, i_w_min += strideW) {

          // Initialize sum.
          float sum = biasW[o_c];

          // For each filter height.
          for (size_t f_h = 0; f_h < kernelH; f_h++) {

            // Skip due to input H padding.
            ssize_t i_h = i_h_min + f_h;
            if (i_h < 0 ||  i_h >= (ssize_t)inWdims[1]) {
              filterW += filterWdims[2] * filterWdims[3];
              continue;
            }

            // For each filter width.
            for (size_t f_w = 0; f_w < kernelW; f_w++) {

              // Skip due to input W padding.
              ssize_t i_w = i_w_min + f_w;
              if (i_w < 0 || i_w >= (ssize_t)inWdims[2]) {
                filterW += filterWdims[3];
                continue;
              }

              // Compute current indices into filter and input buffers.
              size_t inpIdx = i_h * inWdims[2] * inWdims[3] + i_w * inWdims[3];

              // Accumulate along the filter height/width plane.
              sum += (*filterW++) * (inW[inpIdx]);
            }
          }

          // Compute current index into output buffer.
          size_t outIdx = o_h * outWdims[2] * outWdims[3] + o_w * outWdims[3];

          // Write output.
          outW[outIdx] = sum;

          // Reset filter pointer for next output height/width.
          filterW -= filterWdims[1] * filterWdims[2] * filterWdims[3];
        } // W
      } // H

      // Advance filter pointer for next output channel.
      filterW += filterWdims[1] * filterWdims[2] * filterWdims[3];

      // Advance input/output pointer for next output channel.
      inW++;
      outW++;
    } // C

    // Reset filter pointer for next batch.
    filterW -= filterWdims[0] * filterWdims[1] * filterWdims[2] * filterWdims[3];

    // Advance input/output pointers for next batch.
    inW  += inWdims [1] * inWdims [2] * inWdims [3] - inWdims [3];
    outW += outWdims[1] * outWdims[2] * outWdims[3] - outWdims[3];
  } // N
}

#else

// Kernel V2 limitations:
// - dilations are 1
// - group = inpC = outC
void libjit_conv2d_dw_f(float *outW, const float *inW, const float *filterW,
                        const float *biasW, const dim_t *outWdims,
                        const dim_t *inWdims, const dim_t *filterWdims,
                        const dim_t *biasWdims, const dim_t *kernelSizes,
                        const dim_t *strides, const dim_t *pads, dim_t group,
                        unsigned depthUnroll, dim_t dilation) {

  dim_t padT = pads[0];
  dim_t padL = pads[1];

  size_t strideH = strides[0];
  size_t strideW = strides[1];

  size_t kernelH = kernelSizes[0];
  size_t kernelW = kernelSizes[1];

  // For each input in the batch.
  for (size_t n = 0; n < inWdims[0]; n++) {

    // For each output height.
    for (size_t o_h = 0; o_h < outWdims[1]; o_h++) {

      ssize_t f_h_min = MAX(      0,              padT - o_h * strideH);
      ssize_t f_h_max = MIN(kernelH, inWdims[1] + padT - o_h * strideH);

      // For each output width.
      for (size_t o_w = 0; o_w < outWdims[2]; o_w++) {

        ssize_t f_w_min = MAX(      0,              padL - o_w * strideW);
        ssize_t f_w_max = MIN(kernelW, inWdims[2] + padL - o_w * strideW);

        // For each output channel.
        for (size_t o_c = 0; o_c < outWdims[3]; o_c++) {

          // Initialize sum.
          float sum = biasW[o_c];

          // For each filter height.
          for (size_t f_h = f_h_min; f_h < f_h_max; f_h++) {

            // For each filter width.
            for (size_t f_w = f_w_min; f_w < f_w_max; f_w++) {

              // Accumulate along the filter height/width plane.
              sum += (*filterW) * (*inW);

              // Advance pointers for next width.
              filterW ++;
              inW += inWdims[3];
            }
            
            // Advance pointers for next height.
            
            
          }

          // Write output.
          *outW++ = sum;

        } // C
      } // W
    } // H
  } // N
}

#endif

#endif

void libjit_conv2d_i8_i32(
    int8_t *outW, const int8_t *inW, const int8_t *filterW,
    const int32_t *biasW, const dim_t *outWdims, const dim_t *inWdims,
    const dim_t *filterWdims, const dim_t *biasWdims, const dim_t *kernelSizes,
    const dim_t *strides, const dim_t *pads, dim_t group, int32_t outOffset,
    int32_t inOffset, int32_t filterOffset, int32_t biasOffset, int32_t biasPre,
    int32_t biasPost, int32_t biasScale, int32_t outPre, int32_t outPost,
    int32_t outScale, unsigned depthUnroll, dim_t dilation) {
  libjit_quantized_conv2d_generic<int8_t, int32_t>(
      outW, inW, filterW, biasW, outWdims, inWdims, filterWdims, biasWdims,
      kernelSizes, strides, pads, group, outOffset, inOffset, filterOffset,
      biasOffset, biasPre, biasPost, biasScale, outPre, outPost, outScale,
      depthUnroll, dilation);
}

void libjit_conv2d_i8_i8(int8_t *outW, const int8_t *inW, const int8_t *filterW,
                         const int8_t *biasW, const dim_t *outWdims,
                         const dim_t *inWdims, const dim_t *filterWdims,
                         const dim_t *biasWdims, const dim_t *kernelSizes,
                         const dim_t *strides, const dim_t *pads, dim_t group,
                         int32_t outOffset, int32_t inOffset,
                         int32_t filterOffset, int32_t biasOffset,
                         int32_t biasPre, int32_t biasPost, int32_t biasScale,
                         int32_t outPre, int32_t outPost, int32_t outScale,
                         unsigned depthUnroll, dim_t dilation) {
  libjit_quantized_conv2d_generic<int8_t, int8_t>(
      outW, inW, filterW, biasW, outWdims, inWdims, filterWdims, biasWdims,
      kernelSizes, strides, pads, group, outOffset, inOffset, filterOffset,
      biasOffset, biasPre, biasPost, biasScale, outPre, outPost, outScale,
      depthUnroll, dilation);
}

void libjit_channelwise_quantized_conv2d_i8_i32(
    int8_t *outW, const int8_t *inW, const int8_t *filterW,
    const int32_t *biasW, const dim_t *outWdims, const dim_t *inWdims,
    const dim_t *filterWdims, const dim_t *biasWdims, const dim_t *kernels,
    const dim_t *strides, const dim_t *pads, dim_t group, dim_t dilation,
    int32_t outOffset, int32_t inOffset, int32_t *filterOffsetsPtr,
    int32_t *biasOffsetsPtr, const int32_t *biasPrePtr,
    const int32_t *biasPostPtr, const int32_t *biasScalePtr,
    const int32_t *outPrePtr, const int32_t *outPostPtr,
    const int32_t *outScalePtr) {
  libjit_channelwise_quantized_conv2d_generic<int8_t, int32_t>(
      outW, inW, filterW, biasW, outWdims, inWdims, filterWdims, biasWdims,
      kernels, strides, pads, group, dilation, outOffset, inOffset,
      filterOffsetsPtr, biasOffsetsPtr, biasPrePtr, biasPostPtr, biasScalePtr,
      outPrePtr, outPostPtr, outScalePtr);
}

void libjit_channelwise_quantized_conv2d_i8_i8(
    int8_t *outW, const int8_t *inW, const int8_t *filterW, const int8_t *biasW,
    const dim_t *outWdims, const dim_t *inWdims, const dim_t *filterWdims,
    const dim_t *biasWdims, const dim_t *kernels, const dim_t *strides,
    const dim_t *pads, dim_t group, dim_t dilation, int32_t outOffset,
    int32_t inOffset, int32_t *filterOffsetsPtr, int32_t *biasOffsetsPtr,
    const int32_t *biasPrePtr, const int32_t *biasPostPtr,
    const int32_t *biasScalePtr, const int32_t *outPrePtr,
    const int32_t *outPostPtr, const int32_t *outScalePtr) {
  libjit_channelwise_quantized_conv2d_generic<int8_t, int8_t>(
      outW, inW, filterW, biasW, outWdims, inWdims, filterWdims, biasWdims,
      kernels, strides, pads, group, dilation, outOffset, inOffset,
      filterOffsetsPtr, biasOffsetsPtr, biasPrePtr, biasPostPtr, biasScalePtr,
      outPrePtr, outPostPtr, outScalePtr);
}

void libjit_channelwise_quantized_conv3d_i8_i32(
    int8_t *outW, const int8_t *inW, const int8_t *filterW,
    const int32_t *biasW, const dim_t *outWdims, const dim_t *inWdims,
    const dim_t *filterWdims, const dim_t *biasWdims, const dim_t *kernels,
    const dim_t *strides, const dim_t *pads, dim_t group, dim_t dilation,
    int32_t outOffset, int32_t inOffset, int32_t *filterOffsetsPtr,
    int32_t *biasOffsetsPtr, const int32_t *biasPrePtr,
    const int32_t *biasPostPtr, const int32_t *biasScalePtr,
    const int32_t *outPrePtr, const int32_t *outPostPtr,
    const int32_t *outScalePtr) {
  libjit_channelwise_quantized_conv3d_generic<int8_t, int32_t>(
      outW, inW, filterW, biasW, outWdims, inWdims, filterWdims, biasWdims,
      kernels, strides, pads, group, dilation, outOffset, inOffset,
      filterOffsetsPtr, biasOffsetsPtr, biasPrePtr, biasPostPtr, biasScalePtr,
      outPrePtr, outPostPtr, outScalePtr);
}

void libjit_channelwise_quantized_conv3d_i8_i8(
    int8_t *outW, const int8_t *inW, const int8_t *filterW, const int8_t *biasW,
    const dim_t *outWdims, const dim_t *inWdims, const dim_t *filterWdims,
    const dim_t *biasWdims, const dim_t *kernels, const dim_t *strides,
    const dim_t *pads, dim_t group, dim_t dilation, int32_t outOffset,
    int32_t inOffset, int32_t *filterOffsetsPtr, int32_t *biasOffsetsPtr,
    const int32_t *biasPrePtr, const int32_t *biasPostPtr,
    const int32_t *biasScalePtr, const int32_t *outPrePtr,
    const int32_t *outPostPtr, const int32_t *outScalePtr) {
  libjit_channelwise_quantized_conv3d_generic<int8_t, int8_t>(
      outW, inW, filterW, biasW, outWdims, inWdims, filterWdims, biasWdims,
      kernels, strides, pads, group, dilation, outOffset, inOffset,
      filterOffsetsPtr, biasOffsetsPtr, biasPrePtr, biasPostPtr, biasScalePtr,
      outPrePtr, outPostPtr, outScalePtr);
}

void libjit_conv_transpose_f(float *outW, const float *inW,
                             const float *filterW, const float *biasW,
                             const dim_t *outWdims, const dim_t *inWdims,
                             const dim_t *filterWdims, const dim_t *biasWdims,
                             const dim_t *kernels, const dim_t *strides,
                             const dim_t *pads, dim_t group, dim_t dilation) {
  // NHWC format is assumed
  dim_t p = sizeof(float);
  memset(outW, 0, outWdims[0] * outWdims[1] * outWdims[2] * outWdims[3] * p);

  dim_t pad_t = pads[0];
  dim_t pad_l = pads[1];
  dim_t stride_h = strides[0];
  dim_t stride_w = strides[1];
  dim_t kernel_h = kernels[0];
  dim_t kernel_w = kernels[1];
  dim_t outCperG = outWdims[3] / group;
  dim_t inCperG = inWdims[3] / group;

  // For each input in the batch:
  for (dim_t n = 0; n < inWdims[0]; n++) {

    // Initialize the outputs with the bias.
    libjit_conv_init_output_with_bias(n, outW, biasW, outWdims, biasWdims);

    // For each group of input channels:
    for (dim_t g = 0; g < group; g++) {
      for (dim_t d = g * inCperG; d < (g + 1) * inCperG; d++) {
        ssize_t x = -(ssize_t)pad_t;
        for (dim_t bx = 0; bx < inWdims[1]; bx++, x += stride_h) {
          ssize_t y = -(ssize_t)pad_l;
          for (dim_t by = 0; by < inWdims[2]; by++, y += stride_w) {
            float grad = inW[libjit_getXYZW(inWdims, n, bx, by, d)];

            for (dim_t kx = 0; kx < kernel_h; kx++) {
              for (dim_t ky = 0; ky < kernel_w; ky++) {
                ssize_t ax = x + kx * dilation;
                ssize_t ay = y + ky * dilation;

                if (ax < 0 || ay < 0 || ax >= (ssize_t)outWdims[1] ||
                    ay >= (ssize_t)outWdims[2]) {
                  continue;
                }

                for (dim_t c = 0; c < outCperG; c++) {
                  dim_t outIndex = libjit_getXYZW(
                      outWdims, n, (dim_t)ax, (dim_t)ay, (g * outCperG + c));
                  dim_t inIndex = libjit_getXYZW(filterWdims, c, kx, ky, d);
                  outW[outIndex] += filterW[inIndex] * grad;
                }
              }
            }
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}

void libjit_convolution_grad_f(float *inG, const float *outG, const float *inW,
                               float *filterG, float *biasG,
                               const float *filterW, const dim_t *outGdims,
                               const dim_t *inWdims, const dim_t *filterGdims,
                               const dim_t *kernels, const dim_t *strides,
                               const dim_t *pads, dim_t group, dim_t dilation) {
  // NHWC format is assumed
  // Clear inG, filterG, and biasG
  dim_t p = sizeof(float);
  memset(inG, 0, inWdims[0] * inWdims[1] * inWdims[2] * inWdims[3] * p);
  memset(filterG, 0,
         filterGdims[0] * filterGdims[1] * filterGdims[2] * filterGdims[3] * p);
  memset(biasG, 0, outGdims[3] * p);

  dim_t pad_t = pads[0];
  dim_t pad_l = pads[1];
  dim_t stride_h = strides[0];
  dim_t stride_w = strides[1];
  dim_t kernel_h = kernels[0];
  dim_t kernel_w = kernels[1];
  dim_t inCperG = inWdims[3] / group;
  dim_t outCperG = outGdims[3] / group;

  // For each input in the batch:
  for (dim_t n = 0; n < outGdims[0]; n++) {
    // For each group of input channels:
    for (dim_t g = 0; g < group; g++) {
      for (dim_t d = g * outCperG; d < (g + 1) * outCperG; d++) {
        ssize_t x = -(ssize_t)pad_t;
        for (dim_t bx = 0; bx < outGdims[1]; bx++, x += stride_h) {
          ssize_t y = -(ssize_t)pad_l;
          for (dim_t by = 0; by < outGdims[2]; by++, y += stride_w) {
            float grad = outG[libjit_getXYZW(outGdims, n, bx, by, d)];

            for (dim_t kx = 0; kx < kernel_h; kx++) {
              for (dim_t ky = 0; ky < kernel_w; ky++) {
                ssize_t ax = x + kx * dilation;
                ssize_t ay = y + ky * dilation;

                if (ax < 0 || ay < 0 || ax >= (ssize_t)inWdims[1] ||
                    ay >= (ssize_t)inWdims[2]) {
                  continue;
                }

                for (dim_t c = 0; c < inCperG; c++) {
                  inG[libjit_getXYZW(inWdims, n, (dim_t)ax, (dim_t)ay,
                                     g * inCperG + c)] +=
                      filterW[libjit_getXYZW(filterGdims, d, kx, ky, c)] * grad;
                  filterG[libjit_getXYZW(filterGdims, d, kx, ky, c)] +=
                      inW[libjit_getXYZW(inWdims, n, (dim_t)ax, (dim_t)ay,
                                         g * inCperG + c)] *
                      grad;
                }
              }
            }

            biasG[d] += grad;
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}
}
