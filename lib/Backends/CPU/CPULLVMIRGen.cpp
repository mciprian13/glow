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

#include "CPULLVMIRGen.h"

#include "glow/IR/Instrs.h"
#include "glow/LLVMIRCodeGen/LLVMBackend.h"
#include "glow/Quantization/Base/Base.h"

using namespace glow;
using llvm::cast;

CPULLVMIRGen::CPULLVMIRGen(const IRFunction *F,
                           AllocationsInfo &allocationsInfo,
                           std::string mainEntryName, llvm::StringRef libjitBC)
    : LLVMIRGen(F, allocationsInfo, mainEntryName, libjitBC) {}

void CPULLVMIRGen::generateLLVMIRForModule(llvm::IRBuilder<> &builder) {
  // TODO: Add here any backend specific logic.
  LLVMIRGen::generateLLVMIRForModule(builder);
}

void CPULLVMIRGen::generateLLVMIRForInstr(llvm::IRBuilder<> &builder,
                                          const glow::Instruction *I) {
  setCurrentDebugLocation(builder, I);
  assert(!canBePartOfDataParallelKernel(I) &&
         "data parallel instructions are not handled here");

  // ===========================================================================
  //                        ARM Cortex-M Specializations
  // ===========================================================================
  auto targetCPU = getTargetMachine().getTargetCPU().str();
  if (I->getKind() == Kinded::Kind::ConvolutionInstKind &&
      (targetCPU == "cortex-m4" || targetCPU == "cortex-m7" || targetCPU == "cortex-m33")) {
    auto *CI = cast<ConvolutionInst>(I);
    assert(CI->getLayout() == NHWC &&
           "Glow CPU Backend supports only NHWC Convolutions");

    auto *out = CI->getDest();
    auto *inp = CI->getSrc();
    auto *flt = CI->getFilter();
    auto *bias = CI->getBias();

    auto *outTy = out->getType();
    auto *inpTy = inp->getType();
    auto *fltTy = flt->getType();
    auto *biasTy = bias->getType();

    auto *outPtr = emitValueAddress(builder, out);
    auto *inpPtr = emitValueAddress(builder, inp);
    auto *fltPtr = emitValueAddress(builder, flt);
    auto *biasPtr = emitValueAddress(builder, bias);

    if (inpTy->isQuantizedType()) {

      auto *outOffset = emitConstI32(builder, outTy->getOffset());
      auto *inpOffset = emitConstI32(builder, inpTy->getOffset());
      auto *fltOffset = emitConstI32(builder, fltTy->getOffset());

      assert(biasTy->getElementType() == ElemKind::Int32QTy && "Bias precision invalid!");
      assert(biasTy->getOffset() == 0 && "Bias offset must be 0!");
      assert(biasTy->getScale() == inpTy->getScale() * fltTy->getScale() && "Bias scale invalid!");

      // Calculate the scaling parameters for the bias and output.
      float matMulScale = inpTy->getScale() * fltTy->getScale();
      auto outScaleParam = quantization::quantizeScaleOffset32To8(matMulScale / outTy->getScale(), 0);

      auto *outPre = emitConstI32(builder, outScaleParam.pre);
      auto *outPost = emitConstI32(builder, outScaleParam.post);
      auto *outScale = emitConstI32(builder, outScaleParam.scale);

      // ================================= 1x1 Conv =========================================
      if (CI->getKernels()[0] == 1 && CI->getKernels()[1] == 1) {

#if 0
        // Integrate input offset into the bias to reduce run-time overhead.
        Tensor *fltT = getTensorRefForConstantValue(flt);
        Tensor *biasT = getTensorRefForConstantValue(bias);

        auto fltH = fltT->getHandle<int8_t>();
        auto biasH = biasT->getHandle<int32_t>();

        for (size_t idxN = 0; idxN < fltTy->dims()[0]; idxN++) {
          // Compute input offset contribution.
          int64_t sum = 0;
          for (size_t idxH = 0; idxH < fltTy->dims()[1]; idxH++) {
            for (size_t idxW = 0; idxW < fltTy->dims()[2]; idxW++) {
              for (size_t idxC = 0; idxC < fltTy->dims()[3]; idxC++) {
                sum += fltH.at({idxN, idxH, idxW, idxC}) - fltTy->getOffset();
              }
            }
          }
          sum *= -inpTy->getOffset();
          // Add input offset contribution.
          biasH.raw(idxN) = biasH.raw(idxN) + sum;
        }
#endif

        auto *inpRows = emitConstI32(builder, static_cast<int32_t>(inpTy->dims()[1] * inpTy->dims()[2]));
        auto *fltRows = emitConstI32(builder, static_cast<int32_t>(fltTy->dims()[0]));
        auto *numCols = emitConstI32(builder, static_cast<int32_t>(fltTy->dims()[3]));

        auto *F = getFunction("arm_cm_conv2d_1x1_i8xi8");

        createCall(builder, F,
                   {outPtr,
                    inpPtr,
                    fltPtr,
                    biasPtr,
                    inpRows,
                    fltRows,
                    numCols,
                    outOffset,
                    inpOffset,
                    fltOffset,
                    outPre,
                    outPost,
                    outScale
                    });
        std::cout << "Call emitted to arm_cm_conv2d_1x1_i8xi8!\n";
        return;
      }

      // ================================= DW Conv =========================================
      if (CI->getGroup() == inpTy->dims()[3]) {

        auto *outH = emitConstI32(builder, outTy->dims()[1]);
        auto *outW = emitConstI32(builder, outTy->dims()[2]);

        auto *inpN = emitConstI32(builder, inpTy->dims()[0]);
        auto *inpH = emitConstI32(builder, inpTy->dims()[1]);
        auto *inpW = emitConstI32(builder, inpTy->dims()[2]);
        auto *inpC = emitConstI32(builder, inpTy->dims()[3]);

        auto *kernelH = emitConstI32(builder, CI->getKernels()[0]);
        auto *kernelW = emitConstI32(builder, CI->getKernels()[1]);

        auto *strideH = emitConstI32(builder, CI->getStrides()[0]);
        auto *strideW = emitConstI32(builder, CI->getStrides()[1]);

        auto *padT = emitConstI32(builder, CI->getPads()[0]);
        auto *padL = emitConstI32(builder, CI->getPads()[1]);

        auto *F = getFunction("arm_cm_conv2d_dw_i8xi8");

        createCall(builder, F,
                   {outPtr,
                    inpPtr,
                    fltPtr,
                    biasPtr,

                    outH,
                    outW,

                    inpN,
                    inpH,
                    inpW,
                    inpC,

                    kernelH,
                    kernelW,

                    strideH,
                    strideW,

                    padT,
                    padL,

                    outOffset,
                    inpOffset,
                    fltOffset,
                    outPre,
                    outPost,
                    outScale
                    });
        std::cout << "Call emitted to arm_cm_conv2d_dw_i8xi8!\n";
        return;
      }
    }
  }

  // Perform any backend-specific code generation here and delegate everything
  // else to LLVMIRGen.
  switch (I->getKind()) {
  case Kinded::Kind::CPUConvDKKC8InstKind: {
    auto *CI = cast<CPUConvDKKC8Inst>(I);
    auto *dest = CI->getDest();
    auto *src = CI->getSrc();
    auto *filter = CI->getFilter();
    auto *bias = CI->getBias();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);
    auto *filterPtr = emitValueAddress(builder, filter);
    auto *biasPtr = emitValueAddress(builder, bias);

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);
    auto *filterDims = emitValueDims(builder, filter);
    auto *biasDims = emitValueDims(builder, bias);

    auto *kernels = emitConstDimTArray(builder, CI->getKernels());
    auto *strides = emitConstDimTArray(builder, CI->getStrides());
    auto *pads = emitConstDimTArray(builder, CI->getPads());
    auto *group = emitConstDimT(builder, CI->getGroup());

    size_t inChannels = src->dims()[3];
    size_t outChannels = dest->dims()[3];

    // Select a method for iterating on the image in the pixel (filter-first, or
    // input-first). Perform convolutions with a high channel count by scanning
    // the input image multiple times, once for each filter entry. Scan images
    // with a low channel count by scanning the image once because the filter
    // scan will fall in the cache.
    bool pixelScanFirst = (inChannels < 16);

    // The number of float8 registers that we use to process the depth channel.
    unsigned numDepthRegs = (pixelScanFirst ? 8 : 2);
    // The number of y pixels to process at once.
    unsigned sizeGroupY = (pixelScanFirst ? 1 : 5);

    // When producing output pixels process this many times of depth-strips,
    // where each chunk is float8 * numDepthRegs. This is a form of tiling. It's
    // profitable to scan multiple depth-strips of the filter if the scanned
    // memory fits in the cahce and does not get evicted before the next
    // iteration. By increasing the number strips (and using more cache memory)
    // we reduce the number of times that we iterate over the input. However, we
    // also increase the pressure on the cache that has to store the filter so
    // we can't process too many strips at once.
    unsigned depthStrips = 1;
    unsigned stripSize = 8 * numDepthRegs * inChannels;
    unsigned tileSize = 16384;
    // Increase the number of strips until we reach the output-tensor depth size
    // or until we exceed some threashold.
    while (2 * depthStrips * stripSize <= tileSize &&
           2 * depthStrips * numDepthRegs * 8 <= outChannels / CI->getGroup() &&
           depthStrips < 8) {
      depthStrips *= 2;
    }

    auto *pixelScanFirstVal = emitConstI32(builder, pixelScanFirst);
    auto *numDepthRegsVal = emitConstI32(builder, numDepthRegs);
    auto *sizeGroupYVal = emitConstI32(builder, sizeGroupY);
    auto *depthStripsVal = emitConstI32(builder, depthStrips);

    const char *kernelName = "convDKKC8";
    auto *F = getFunction(kernelName, dest->getElementType());

    createCall(builder, F,
               {destPtr, srcPtr, filterPtr, biasPtr, destDims, srcDims,
                filterDims, biasDims, kernels, strides, pads, group,
                pixelScanFirstVal, numDepthRegsVal, sizeGroupYVal,
                depthStripsVal});
    break;
  }
  default:
    LLVMIRGen::generateLLVMIRForInstr(builder, I);
  }
}

void CPULLVMIRGen::generateLLVMIRForDataParallelInstr(
    llvm::IRBuilder<> &builder, const glow::Instruction *I,
    llvm::Function *kernel, llvm::DenseMap<Value *, int> &bufferToArgNum,
    llvm::Value *loopCount) {
  setCurrentDebugLocation(builder, I);
  assert(canBePartOfDataParallelKernel(I) &&
         "Expected a data parallel instruction");
  // Perform any backend-specific code generation here and delegate everything
  // else to LLVMIRGen.
  switch (I->getKind()) {
  case Kinded::Kind::CPUMaxSplatInstKind: {
    auto *AN = cast<CPUMaxSplatInst>(I);
    auto *dest = AN->getDest();
    auto V = AN->getSplatValue();
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *lhs = AN->getSrc();
    auto *lhsPtr = emitBufferAddress(builder, lhs, kernel, bufferToArgNum);
    auto *F = getFunction("element_maxsplat_kernel", dest->getElementType());
    auto *elementTy = getElementType(builder, dest);
    auto *pointerNull =
        llvm::ConstantPointerNull::get(elementTy->getPointerTo());

    if (lhs->getType()->isQuantizedType()) {
      // Quantize value from the splat to the {S,O} of the lhs param.
      TensorQuantizationParams TQP{lhs->getType()->getScale(),
                                   lhs->getType()->getOffset()};
      auto quantizedValue = quantization::quantize(V, TQP);
      auto *val = emitConst(builder, quantizedValue, lhs->getElementType());
      auto *stackedOpCall = createUncheckedCall(
          builder, F, {loopCount, val, lhsPtr, pointerNull});
      auto *destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    } else {
      auto *val = emitConst(builder, V, lhs->getElementType());
      auto *stackedOpCall = createUncheckedCall(
          builder, F, {loopCount, val, lhsPtr, pointerNull});
      auto *destAddr = builder.CreateGEP(builder.getFloatTy(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    }

    break;
  }

  default:
    LLVMIRGen::generateLLVMIRForDataParallelInstr(builder, I, kernel,
                                                  bufferToArgNum, loopCount);
  }
}
