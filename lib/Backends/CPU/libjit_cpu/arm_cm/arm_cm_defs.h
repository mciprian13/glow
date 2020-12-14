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
#ifndef GLOW_LLVMIRCODEGEN_LIBJIT_ARM_CM_DEFS_H
#define GLOW_LLVMIRCODEGEN_LIBJIT_ARM_CM_DEFS_H

//#include <assert.h>
//#include <cmath>
//#include <cstdlib>
#include <math.h>
#include <stdint.h>
//#include <string.h>

//#include "../../../../LLVMIRCodeGen/libjit/libjit_defs.h"

#define __ASM __asm

#define __STATIC_FORCEINLINE  __attribute__((always_inline)) static inline

__STATIC_FORCEINLINE
int32_t arm_cm_scale_i32i8(int32_t input, int32_t pre, int32_t post,
                           int32_t scale, int32_t offset) {
  // The operation x >> post is rounded down to negative infinity. To get to
  // round-nearest we add (1 << (post - 1)) to the value prior to shifting.
  // Rounding is performed only when shifting right (pos > 0).
  int rtn = (post > 0) ? (1 << (post - 1)) : 0;

  // NOTICE: If your tests are failing because of signed integer overflow then
  // this is a bug in the test and not in the program. You should make sure that
  // the inputs to the operations do not overflow. The semantics of the
  // quantization process is such that the result for values that fall out of
  // range is undefined. The conversion procedure will only tolerate a few bits
  // of overflow and the result will be clipped.
  return ((((input >> pre) * scale) + rtn) >> post) + offset;
}

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

//===----------------------------------------------------------------------===//
//                           ARM Cortex-M Intrinsics
//===----------------------------------------------------------------------===//
__STATIC_FORCEINLINE uint32_t __SSAT8(uint32_t op)
{
  uint32_t result;
  __ASM("ssat %0, #8, %1" : "=r" (result) : "r" (op));
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SXTB16_ROR0(uint32_t op1)
{
  uint32_t result;
  __ASM("sxtb16 %0, %1, ROR #0" : "=r" (result) : "r" (op1));
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SXTB16_ROR8(uint32_t op1)
{
  uint32_t result;
  __ASM("sxtb16 %0, %1, ROR #8" : "=r" (result) : "r" (op1));
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SXTAB16_ROR0(uint32_t op1, uint32_t op2)
{
  uint32_t result;
  __ASM("sxtab16 %0, %1, %2, ROR #0" : "=r" (result) : "r" (op1), "r" (op2));
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SXTAB16_ROR8(uint32_t op1, uint32_t op2)
{
  uint32_t result;
  __ASM("sxtab16 %0, %1, %2, ROR #8" : "=r" (result) : "r" (op1), "r" (op2));
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SMLAD(uint32_t op1, uint32_t op2, uint32_t op3)
{
  uint32_t result;
  __ASM volatile ("smlad %0, %1, %2, %3" : "=r" (result) : "r" (op1), "r" (op2), "r" (op3) );
  return(result);
}

#define __PKHBT(ARG1,ARG2,ARG3) \
({                          \
  uint32_t __RES, __ARG1 = (ARG1), __ARG2 = (ARG2); \
  __ASM ("pkhbt %0, %1, %2, lsl %3" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2), "I" (ARG3)  ); \
  __RES; \
})

__STATIC_FORCEINLINE uint32_t __LDR_INC(const int8_t **addr)
{
  uint32_t result;
  result = *(const uint32_t *)(*addr);
  *addr += 4;
  return result;
}

__STATIC_FORCEINLINE uint32_t __LDR(const int8_t *addr)
{
  return *(const uint32_t *)(addr);
}

//===----------------------------------------------------------------------===//
//                           ARM Cortex-M Kernels
//===----------------------------------------------------------------------===//
// Conclusions:
// - Differentiating all the pointers we add a slight overhead:
//   e.g. 251 ms vs 247 ms for test suite.
// ----------------------------------------------------------------------------
// - Without LHS offset compensation (inlined): 121 ms for MobileNet v1 0.25
// - With    LHS offset compensation (inlined): 107 ms for MobileNet v1 0.25 (112 ms if done @ compile-time)
// ----------------------------------------------------------------------------
// - Without LHS offset compensation (no-inline): 122 ms for MobileNet v1 0.25
// - With    LHS offset compensation (no-inline): 109 ms for MobileNet v1 0.25
// ----------------------------------------------------------------------------
// Compiled externally with GCC:
// - Without LHS offset compensation (no-inline):  96 ms for MobileNet v1 0.25
// - With    LHS offset compensation (no-inline):  95 ms for MobileNet v1 0.25
// ----------------------------------------------------------------------------
// Reference for initial kernel used to obtain this:
// - main loop:      89 cycles
// - secondary loop: 57 cycles
// ----------------------------------------------------------------------------
// NOTE: We can get better performance if we could use 0 RHS offset.
// ----------------------------------------------------------------------------

//__STATIC_FORCEINLINE
void arm_cm_mat_mul_rhst_i8xi8(const int8_t *lhsPtr,
                               const int8_t *rhsPtr,
                               const int32_t *biasPtr,
                                      int8_t *outPtr,
                               const int32_t lhsRows,
                               const int32_t rhsRows,
                               const int32_t numCols,
                               const int32_t lhsOffset,
                               const int32_t rhsOffset,
                               const int32_t outOffset,

                               const int32_t outPre,
                               const int32_t outPost,
                               const int32_t outScale,

                               const int32_t *outMultipliers,
                               const int32_t *outShifters);
#if 0
{
    #define PRE_LHS_OFFSET 1

    const int32_t rowOff = numCols - 4;

    // Pack LHS and RHS offsets as 2 x int16.
    // We invert the sign sign since these are used for addition.
#if !PRE_LHS_OFFSET
    const int32_t lhsOffx2 = __PKHBT(-lhsOffset, -lhsOffset, 16);
#endif
    const int32_t rhsOffx2 = __PKHBT(-rhsOffset, -rhsOffset, 16);

    // Iterate 2 RHS rows at once.
    for (int32_t rhsRowIdx = 0; rhsRowIdx <= (rhsRows - 2); rhsRowIdx += 2)
    {
        const int8_t *lhsAddr = lhsPtr;

#if PRE_LHS_OFFSET
        // Pre-compensate the LHS offset by incorporating in bias.
        int32_t bias0 = 0;
        int32_t bias1 = 0;
        for (int32_t idx = 0; idx < numCols; ++idx)
        {
            bias0 += rhsPtr[idx] - rhsOffset;
            bias1 += rhsPtr[idx + numCols] - rhsOffset;
        }
        bias0 *= -lhsOffset;
        bias1 *= -lhsOffset;

        bias0 += biasPtr[rhsRowIdx + 0];
        bias1 += biasPtr[rhsRowIdx + 1];

#else

        int32_t bias0 = biasPtr[rhsRowIdx + 0];
        int32_t bias1 = biasPtr[rhsRowIdx + 1];

#endif

        // Iterate 2 LHS rows at once.
        for (int32_t lhsRowIdx = (lhsRows >> 1); lhsRowIdx > 0; --lhsRowIdx)
        {
            const int8_t *rhsAddr = rhsPtr;

            int32_t out00 = bias0;
            int32_t out01 = bias1;
            int32_t out10 = bias0;
            int32_t out11 = bias1;

            int32_t reg0, reg1, reg2, reg3, reg4, reg5;

            // Perform dot-product for all LHS and RHS columns.
            int32_t colIdx = 0;
            for (; colIdx <= (numCols - 16); colIdx += 16)
            {
#if PRE_LHS_OFFSET
                // Load 4 x LHS and 4 x RHS.
                reg0 = __LDR_INC(&lhsAddr);
                reg1 = __SXTB16_ROR0(reg0);
                reg0 = __SXTB16_ROR8(reg0);
                reg2 = __LDR_INC(&rhsAddr);
                reg3 = __SXTAB16_ROR0(rhsOffx2, reg2);
                reg2 = __SXTAB16_ROR8(rhsOffx2, reg2);

                // 4 x MAC out00 and out01.
                reg4 = __LDR(rhsAddr + rowOff);
                reg5 = __SXTAB16_ROR0(rhsOffx2, reg4);
                reg4 = __SXTAB16_ROR8(rhsOffx2, reg4);
                out00 = __SMLAD(reg0, reg2, out00);
                out00 = __SMLAD(reg1, reg3, out00);
                out01 = __SMLAD(reg0, reg4, out01);
                out01 = __SMLAD(reg1, reg5, out01);

                // 4 x MAC out10 and out11.
                reg0 = __LDR(lhsAddr + rowOff);
                reg1 = __SXTB16_ROR0(reg0);
                reg0 = __SXTB16_ROR8(reg0);
                out10 = __SMLAD(reg0, reg2, out10);
                out10 = __SMLAD(reg1, reg3, out10);
                out11 = __SMLAD(reg0, reg4, out11);
                out11 = __SMLAD(reg1, reg5, out11);

                // Load 4 x LHS and 4 x RHS.
                reg0 = __LDR_INC(&lhsAddr);
                reg1 = __SXTB16_ROR0(reg0);
                reg0 = __SXTB16_ROR8(reg0);
                reg2 = __LDR_INC(&rhsAddr);
                reg3 = __SXTAB16_ROR0(rhsOffx2, reg2);
                reg2 = __SXTAB16_ROR8(rhsOffx2, reg2);

                // 4 x MAC out00 and out01.
                reg4 = __LDR(rhsAddr + rowOff);
                reg5 = __SXTAB16_ROR0(rhsOffx2, reg4);
                reg4 = __SXTAB16_ROR8(rhsOffx2, reg4);
                out00 = __SMLAD(reg0, reg2, out00);
                out00 = __SMLAD(reg1, reg3, out00);
                out01 = __SMLAD(reg0, reg4, out01);
                out01 = __SMLAD(reg1, reg5, out01);

                // 4 x MAC out10 and out11.
                reg0 = __LDR(lhsAddr + rowOff);
                reg1 = __SXTB16_ROR0(reg0);
                reg0 = __SXTB16_ROR8(reg0);
                out10 = __SMLAD(reg0, reg2, out10);
                out10 = __SMLAD(reg1, reg3, out10);
                out11 = __SMLAD(reg0, reg4, out11);
                out11 = __SMLAD(reg1, reg5, out11);

                // Load 4 x LHS and 4 x RHS.
                reg0 = __LDR_INC(&lhsAddr);
                reg1 = __SXTB16_ROR0(reg0);
                reg0 = __SXTB16_ROR8(reg0);
                reg2 = __LDR_INC(&rhsAddr);
                reg3 = __SXTAB16_ROR0(rhsOffx2, reg2);
                reg2 = __SXTAB16_ROR8(rhsOffx2, reg2);

                // 4 x MAC out00 and out01.
                reg4 = __LDR(rhsAddr + rowOff);
                reg5 = __SXTAB16_ROR0(rhsOffx2, reg4);
                reg4 = __SXTAB16_ROR8(rhsOffx2, reg4);
                out00 = __SMLAD(reg0, reg2, out00);
                out00 = __SMLAD(reg1, reg3, out00);
                out01 = __SMLAD(reg0, reg4, out01);
                out01 = __SMLAD(reg1, reg5, out01);

                // 4 x MAC out10 and out11.
                reg0 = __LDR(lhsAddr + rowOff);
                reg1 = __SXTB16_ROR0(reg0);
                reg0 = __SXTB16_ROR8(reg0);
                out10 = __SMLAD(reg0, reg2, out10);
                out10 = __SMLAD(reg1, reg3, out10);
                out11 = __SMLAD(reg0, reg4, out11);
                out11 = __SMLAD(reg1, reg5, out11);

                // Load 4 x LHS and 4 x RHS.
                reg0 = __LDR_INC(&lhsAddr);
                reg1 = __SXTB16_ROR0(reg0);
                reg0 = __SXTB16_ROR8(reg0);
                reg2 = __LDR_INC(&rhsAddr);
                reg3 = __SXTAB16_ROR0(rhsOffx2, reg2);
                reg2 = __SXTAB16_ROR8(rhsOffx2, reg2);

                // 4 x MAC out00 and out01.
                reg4 = __LDR(rhsAddr + rowOff);
                reg5 = __SXTAB16_ROR0(rhsOffx2, reg4);
                reg4 = __SXTAB16_ROR8(rhsOffx2, reg4);
                out00 = __SMLAD(reg0, reg2, out00);
                out00 = __SMLAD(reg1, reg3, out00);
                out01 = __SMLAD(reg0, reg4, out01);
                out01 = __SMLAD(reg1, reg5, out01);

                // 4 x MAC out10 and out11.
                reg0 = __LDR(lhsAddr + rowOff);
                reg1 = __SXTB16_ROR0(reg0);
                reg0 = __SXTB16_ROR8(reg0);
                out10 = __SMLAD(reg0, reg2, out10);
                out10 = __SMLAD(reg1, reg3, out10);
                out11 = __SMLAD(reg0, reg4, out11);
                out11 = __SMLAD(reg1, reg5, out11);
#else
                // Load 4 x LHS and 4 x RHS.
                reg0 = __LDR_INC(&lhsAddr);
                reg1 = __SXTAB16_ROR0(lhsOffx2, reg0);
                reg0 = __SXTAB16_ROR8(lhsOffx2, reg0);
                reg2 = __LDR_INC(&rhsAddr);
                reg3 = __SXTAB16_ROR0(rhsOffx2, reg2);
                reg2 = __SXTAB16_ROR8(rhsOffx2, reg2);

                // 4 x MAC out00 and out01.
                reg4 = __LDR(rhsAddr + rowOff);
                reg5 = __SXTAB16_ROR0(rhsOffx2, reg4);
                reg4 = __SXTAB16_ROR8(rhsOffx2, reg4);
                out00 = __SMLAD(reg0, reg2, out00);
                out00 = __SMLAD(reg1, reg3, out00);
                out01 = __SMLAD(reg0, reg4, out01);
                out01 = __SMLAD(reg1, reg5, out01);

                // 4 x MAC out10 and out11.
                reg0 = __LDR(lhsAddr + rowOff);
                reg1 = __SXTAB16_ROR0(lhsOffx2, reg0);
                reg0 = __SXTAB16_ROR8(lhsOffx2, reg0);
                out10 = __SMLAD(reg0, reg2, out10);
                out10 = __SMLAD(reg1, reg3, out10);
                out11 = __SMLAD(reg0, reg4, out11);
                out11 = __SMLAD(reg1, reg5, out11);

                // Load 4 x LHS and 4 x RHS.
                reg0 = __LDR_INC(&lhsAddr);
                reg1 = __SXTAB16_ROR0(lhsOffx2, reg0);
                reg0 = __SXTAB16_ROR8(lhsOffx2, reg0);
                reg2 = __LDR_INC(&rhsAddr);
                reg3 = __SXTAB16_ROR0(rhsOffx2, reg2);
                reg2 = __SXTAB16_ROR8(rhsOffx2, reg2);

                // 4 x MAC out00 and out01.
                reg4 = __LDR(rhsAddr + rowOff);
                reg5 = __SXTAB16_ROR0(rhsOffx2, reg4);
                reg4 = __SXTAB16_ROR8(rhsOffx2, reg4);
                out00 = __SMLAD(reg0, reg2, out00);
                out00 = __SMLAD(reg1, reg3, out00);
                out01 = __SMLAD(reg0, reg4, out01);
                out01 = __SMLAD(reg1, reg5, out01);

                // 4 x MAC out10 and out11.
                reg0 = __LDR(lhsAddr + rowOff);
                reg1 = __SXTAB16_ROR0(lhsOffx2, reg0);
                reg0 = __SXTAB16_ROR8(lhsOffx2, reg0);
                out10 = __SMLAD(reg0, reg2, out10);
                out10 = __SMLAD(reg1, reg3, out10);
                out11 = __SMLAD(reg0, reg4, out11);
                out11 = __SMLAD(reg1, reg5, out11);

                // Load 4 x LHS and 4 x RHS.
                reg0 = __LDR_INC(&lhsAddr);
                reg1 = __SXTAB16_ROR0(lhsOffx2, reg0);
                reg0 = __SXTAB16_ROR8(lhsOffx2, reg0);
                reg2 = __LDR_INC(&rhsAddr);
                reg3 = __SXTAB16_ROR0(rhsOffx2, reg2);
                reg2 = __SXTAB16_ROR8(rhsOffx2, reg2);

                // 4 x MAC out00 and out01.
                reg4 = __LDR(rhsAddr + rowOff);
                reg5 = __SXTAB16_ROR0(rhsOffx2, reg4);
                reg4 = __SXTAB16_ROR8(rhsOffx2, reg4);
                out00 = __SMLAD(reg0, reg2, out00);
                out00 = __SMLAD(reg1, reg3, out00);
                out01 = __SMLAD(reg0, reg4, out01);
                out01 = __SMLAD(reg1, reg5, out01);

                // 4 x MAC out10 and out11.
                reg0 = __LDR(lhsAddr + rowOff);
                reg1 = __SXTAB16_ROR0(lhsOffx2, reg0);
                reg0 = __SXTAB16_ROR8(lhsOffx2, reg0);
                out10 = __SMLAD(reg0, reg2, out10);
                out10 = __SMLAD(reg1, reg3, out10);
                out11 = __SMLAD(reg0, reg4, out11);
                out11 = __SMLAD(reg1, reg5, out11);

                // Load 4 x LHS and 4 x RHS.
                reg0 = __LDR_INC(&lhsAddr);
                reg1 = __SXTAB16_ROR0(lhsOffx2, reg0);
                reg0 = __SXTAB16_ROR8(lhsOffx2, reg0);
                reg2 = __LDR_INC(&rhsAddr);
                reg3 = __SXTAB16_ROR0(rhsOffx2, reg2);
                reg2 = __SXTAB16_ROR8(rhsOffx2, reg2);

                // 4 x MAC out00 and out01.
                reg4 = __LDR(rhsAddr + rowOff);
                reg5 = __SXTAB16_ROR0(rhsOffx2, reg4);
                reg4 = __SXTAB16_ROR8(rhsOffx2, reg4);
                out00 = __SMLAD(reg0, reg2, out00);
                out00 = __SMLAD(reg1, reg3, out00);
                out01 = __SMLAD(reg0, reg4, out01);
                out01 = __SMLAD(reg1, reg5, out01);

                // 4 x MAC out10 and out11.
                reg0 = __LDR(lhsAddr + rowOff);
                reg1 = __SXTAB16_ROR0(lhsOffx2, reg0);
                reg0 = __SXTAB16_ROR8(lhsOffx2, reg0);
                out10 = __SMLAD(reg0, reg2, out10);
                out10 = __SMLAD(reg1, reg3, out10);
                out11 = __SMLAD(reg0, reg4, out11);
                out11 = __SMLAD(reg1, reg5, out11);
#endif
            }

            // Remaining columns.
            for (; colIdx < numCols; ++colIdx)
            {
#if PRE_LHS_OFFSET
                int32_t rhsVal0 = rhsAddr[0] - rhsOffset;
                int32_t rhsVal1 = rhsAddr[numCols] - rhsOffset;

                int32_t lhsVal  = lhsAddr[0];
                out00 += lhsVal * rhsVal0;
                out01 += lhsVal * rhsVal1;

                lhsVal  = lhsAddr[numCols];
                out10 += lhsVal * rhsVal0;
                out11 += lhsVal * rhsVal1;

                ++rhsAddr;
                ++lhsAddr;
#else
                int32_t rhsVal0 = rhsAddr[0] - rhsOffset;
                int32_t rhsVal1 = rhsAddr[numCols] - rhsOffset;

                int32_t lhsVal  = lhsAddr[0] - lhsOffset;
                out00 += lhsVal * rhsVal0;
                out01 += lhsVal * rhsVal1;

                lhsVal  = lhsAddr[numCols] - lhsOffset;
                out10 += lhsVal * rhsVal0;
                out11 += lhsVal * rhsVal1;

                ++rhsAddr;
                ++lhsAddr;
#endif
            }

            // Scale to int8.
            out00 = arm_cm_scale_i32i8(out00, outPre, outPost, outScale, outOffset);
            out01 = arm_cm_scale_i32i8(out01, outPre, outPost, outScale, outOffset);
            out10 = arm_cm_scale_i32i8(out10, outPre, outPost, outScale, outOffset);
            out11 = arm_cm_scale_i32i8(out11, outPre, outPost, outScale, outOffset);

            // Saturate to int8.
            out00 = __SSAT8(out00);
            out01 = __SSAT8(out01);
            out10 = __SSAT8(out10);
            out11 = __SSAT8(out11);

            *outPtr++ = out00;
            *outPtr++ = out01;
            outPtr += rhsRows - 2;

            *outPtr++ = out10;
            *outPtr++ = out11;
            outPtr += rhsRows - 2;

            lhsAddr += numCols;
        }

        // Remaining LHS rows.
        if (lhsRows % 2)
        {
            const int8_t *rhsAddr = rhsPtr;

            int32_t out00 = bias0;
            int32_t out01 = bias1;

            int32_t reg0, reg1, reg2, reg3, reg4, reg5;

            // Perform dot-product for all LHS and RHS columns.
            int32_t colIdx = 0;
            for (; colIdx <= (numCols - 16); colIdx += 16)
            {
#if PRE_LHS_OFFSET
                // Load 4 x LHS and 4 x RHS.
                reg0 = __LDR_INC(&lhsAddr);
                reg1 = __SXTB16_ROR0(reg0);
                reg0 = __SXTB16_ROR8(reg0);
                reg2 = __LDR_INC(&rhsAddr);
                reg3 = __SXTAB16_ROR0(rhsOffx2, reg2);
                reg2 = __SXTAB16_ROR8(rhsOffx2, reg2);

                // 4 x MAC out00 and out01.
                reg4 = __LDR(rhsAddr + rowOff);
                reg5 = __SXTAB16_ROR0(rhsOffx2, reg4);
                reg4 = __SXTAB16_ROR8(rhsOffx2, reg4);
                out00 = __SMLAD(reg0, reg2, out00);
                out00 = __SMLAD(reg1, reg3, out00);
                out01 = __SMLAD(reg0, reg4, out01);
                out01 = __SMLAD(reg1, reg5, out01);

                // Load 4 x LHS and 4 x RHS.
                reg0 = __LDR_INC(&lhsAddr);
                reg1 = __SXTB16_ROR0(reg0);
                reg0 = __SXTB16_ROR8(reg0);
                reg2 = __LDR_INC(&rhsAddr);
                reg3 = __SXTAB16_ROR0(rhsOffx2, reg2);
                reg2 = __SXTAB16_ROR8(rhsOffx2, reg2);

                // 4 x MAC out00 and out01.
                reg4 = __LDR(rhsAddr + rowOff);
                reg5 = __SXTAB16_ROR0(rhsOffx2, reg4);
                reg4 = __SXTAB16_ROR8(rhsOffx2, reg4);
                out00 = __SMLAD(reg0, reg2, out00);
                out00 = __SMLAD(reg1, reg3, out00);
                out01 = __SMLAD(reg0, reg4, out01);
                out01 = __SMLAD(reg1, reg5, out01);

                // Load 4 x LHS and 4 x RHS.
                reg0 = __LDR_INC(&lhsAddr);
                reg1 = __SXTB16_ROR0(reg0);
                reg0 = __SXTB16_ROR8(reg0);
                reg2 = __LDR_INC(&rhsAddr);
                reg3 = __SXTAB16_ROR0(rhsOffx2, reg2);
                reg2 = __SXTAB16_ROR8(rhsOffx2, reg2);

                // 4 x MAC out00 and out01.
                reg4 = __LDR(rhsAddr + rowOff);
                reg5 = __SXTAB16_ROR0(rhsOffx2, reg4);
                reg4 = __SXTAB16_ROR8(rhsOffx2, reg4);
                out00 = __SMLAD(reg0, reg2, out00);
                out00 = __SMLAD(reg1, reg3, out00);
                out01 = __SMLAD(reg0, reg4, out01);
                out01 = __SMLAD(reg1, reg5, out01);

                // Load 4 x LHS and 4 x RHS.
                reg0 = __LDR_INC(&lhsAddr);
                reg1 = __SXTB16_ROR0(reg0);
                reg0 = __SXTB16_ROR8(reg0);
                reg2 = __LDR_INC(&rhsAddr);
                reg3 = __SXTAB16_ROR0(rhsOffx2, reg2);
                reg2 = __SXTAB16_ROR8(rhsOffx2, reg2);

                // 4 x MAC out00 and out01.
                reg4 = __LDR(rhsAddr + rowOff);
                reg5 = __SXTAB16_ROR0(rhsOffx2, reg4);
                reg4 = __SXTAB16_ROR8(rhsOffx2, reg4);
                out00 = __SMLAD(reg0, reg2, out00);
                out00 = __SMLAD(reg1, reg3, out00);
                out01 = __SMLAD(reg0, reg4, out01);
                out01 = __SMLAD(reg1, reg5, out01);
#else
                // Load 4 x LHS and 4 x RHS.
                reg0 = __LDR_INC(&lhsAddr);
                reg1 = __SXTAB16_ROR0(lhsOffx2, reg0);
                reg0 = __SXTAB16_ROR8(lhsOffx2, reg0);
                reg2 = __LDR_INC(&rhsAddr);
                reg3 = __SXTAB16_ROR0(rhsOffx2, reg2);
                reg2 = __SXTAB16_ROR8(rhsOffx2, reg2);

                // 4 x MAC out00 and out01.
                reg4 = __LDR(rhsAddr + rowOff);
                reg5 = __SXTAB16_ROR0(rhsOffx2, reg4);
                reg4 = __SXTAB16_ROR8(rhsOffx2, reg4);
                out00 = __SMLAD(reg0, reg2, out00);
                out00 = __SMLAD(reg1, reg3, out00);
                out01 = __SMLAD(reg0, reg4, out01);
                out01 = __SMLAD(reg1, reg5, out01);

                // Load 4 x LHS and 4 x RHS.
                reg0 = __LDR_INC(&lhsAddr);
                reg1 = __SXTAB16_ROR0(lhsOffx2, reg0);
                reg0 = __SXTAB16_ROR8(lhsOffx2, reg0);
                reg2 = __LDR_INC(&rhsAddr);
                reg3 = __SXTAB16_ROR0(rhsOffx2, reg2);
                reg2 = __SXTAB16_ROR8(rhsOffx2, reg2);

                // 4 x MAC out00 and out01.
                reg4 = __LDR(rhsAddr + rowOff);
                reg5 = __SXTAB16_ROR0(rhsOffx2, reg4);
                reg4 = __SXTAB16_ROR8(rhsOffx2, reg4);
                out00 = __SMLAD(reg0, reg2, out00);
                out00 = __SMLAD(reg1, reg3, out00);
                out01 = __SMLAD(reg0, reg4, out01);
                out01 = __SMLAD(reg1, reg5, out01);

                // Load 4 x LHS and 4 x RHS.
                reg0 = __LDR_INC(&lhsAddr);
                reg1 = __SXTAB16_ROR0(lhsOffx2, reg0);
                reg0 = __SXTAB16_ROR8(lhsOffx2, reg0);
                reg2 = __LDR_INC(&rhsAddr);
                reg3 = __SXTAB16_ROR0(rhsOffx2, reg2);
                reg2 = __SXTAB16_ROR8(rhsOffx2, reg2);

                // 4 x MAC out00 and out01.
                reg4 = __LDR(rhsAddr + rowOff);
                reg5 = __SXTAB16_ROR0(rhsOffx2, reg4);
                reg4 = __SXTAB16_ROR8(rhsOffx2, reg4);
                out00 = __SMLAD(reg0, reg2, out00);
                out00 = __SMLAD(reg1, reg3, out00);
                out01 = __SMLAD(reg0, reg4, out01);
                out01 = __SMLAD(reg1, reg5, out01);

                // Load 4 x LHS and 4 x RHS.
                reg0 = __LDR_INC(&lhsAddr);
                reg1 = __SXTAB16_ROR0(lhsOffx2, reg0);
                reg0 = __SXTAB16_ROR8(lhsOffx2, reg0);
                reg2 = __LDR_INC(&rhsAddr);
                reg3 = __SXTAB16_ROR0(rhsOffx2, reg2);
                reg2 = __SXTAB16_ROR8(rhsOffx2, reg2);

                // 4 x MAC out00 and out01.
                reg4 = __LDR(rhsAddr + rowOff);
                reg5 = __SXTAB16_ROR0(rhsOffx2, reg4);
                reg4 = __SXTAB16_ROR8(rhsOffx2, reg4);
                out00 = __SMLAD(reg0, reg2, out00);
                out00 = __SMLAD(reg1, reg3, out00);
                out01 = __SMLAD(reg0, reg4, out01);
                out01 = __SMLAD(reg1, reg5, out01);
#endif
            }

            // Remaining columns.
            for (; colIdx < numCols; ++colIdx)
            {
#if PRE_LHS_OFFSET
                int32_t lhsVal = lhsAddr[0];
                int32_t rhsVal0 = rhsAddr[0] - rhsOffset;
                int32_t rhsVal1 = rhsAddr[numCols] - rhsOffset;
                out00 += lhsVal * rhsVal0;
                out01 += lhsVal * rhsVal1;
                ++rhsAddr;
                ++lhsAddr;
#else
                int32_t lhsVal = lhsAddr[0] - lhsOffset;
                int32_t rhsVal0 = rhsAddr[0] - rhsOffset;
                int32_t rhsVal1 = rhsAddr[numCols] - rhsOffset;
                out00 += lhsVal * rhsVal0;
                out01 += lhsVal * rhsVal1;
                ++rhsAddr;
                ++lhsAddr;
#endif
            }

            // Scale to int8.
            out00 = arm_cm_scale_i32i8(out00, outPre, outPost, outScale, outOffset);
            out01 = arm_cm_scale_i32i8(out01, outPre, outPost, outScale, outOffset);

            // Saturate to int8.
            out00 = __SSAT8(out00);
            out01 = __SSAT8(out01);

            *outPtr++ = out00;
            *outPtr++ = out01;
            outPtr += rhsRows - 2;
        }

        rhsPtr += 2 * numCols;
        outPtr -= rhsRows * lhsRows - 2;
    }

    // Remaining RHS rows.
    if (rhsRows % 2)
    {
        const int8_t *lhsAddr = lhsPtr;

        for (int32_t lhsRowIdx = 0; lhsRowIdx < lhsRows; ++lhsRowIdx)
        {
            const int8_t *rhsAddr = rhsPtr;

            int32_t out00 = biasPtr[rhsRows - 1];

            for (int32_t colIdx = 0; colIdx < numCols; ++colIdx)
            {
                int32_t lhsVal = lhsAddr[0] - lhsOffset;
                int32_t rhsVal = rhsAddr[0] - rhsOffset;
                out00 += lhsVal * rhsVal;
                ++rhsAddr;
                ++lhsAddr;
            }

            // Scale to int8.
            out00 = arm_cm_scale_i32i8(out00, outPre, outPost, outScale, outOffset);

            // Saturate to int8.
            out00 = __SSAT8(out00);

            *outPtr = out00;
            outPtr += rhsRows;
        }
    }
}
#endif

#endif // GLOW_LLVMIRCODEGEN_LIBJIT_ARM_CM_DEFS_H
