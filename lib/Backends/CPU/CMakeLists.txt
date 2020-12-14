add_subdirectory(ClassGen)

include(GlowAddLibjit)

# LIBJIT LLVM common source files.
set(LIBJIT_CPU_SOURCE_FILES)
list(APPEND LIBJIT_CPU_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../LLVMIRCodeGen/libjit/libjit.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../LLVMIRCodeGen/libjit/libjit_conv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../LLVMIRCodeGen/libjit/libjit_matmul.cpp
)

# LIBJIT CPU specific source files.
list(APPEND LIBJIT_CPU_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/libjit_cpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/libjit_cpu_conv.cpp
)

# LIBJIT CPU compile options.
set(LIBJIT_CPU_COMPILE_OPTIONS
  -I${CMAKE_CURRENT_SOURCE_DIR}/../../LLVMIRCodeGen/libjit
  -std=c++14
  -g
  -O0

  # When building for a custom target, use this as an example of how to
  # set up cross-compilation correctly. Then, when building bundles,
  # do not forget to specify the -target= option with the same target
  # string. Also, when building bundles, specify the -llvm-compiler option
  # together with -llvm-compiler-opt to specify custom compilation flags for
  # cross-compilation.

  # --sysroot=/usr/arm-linux-gnueabihf/
  # -target armv7-neon-linux-gnueabihf
  # -I/usr/arm-linux-gnueabihf/include/c++/7.4.0/arm-linux-gnueabihf/
  ${LLVMCPURuntimeExtraFlags}
)

# LIBJIT CPU specific object files.
set(LIBJIT_CPU_OBJECT_FILES)

# LIBJIT CPU specific dependencies.
set(LIBJIT_CPU_DEPENDENCIES)

#===============================================================================
#                                  ARM Cortex-M
#===============================================================================
# ARM CM specific source files.
list(APPEND ARM_CM_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/arm_cm/arm_cm.c
)

# ARM CM compile options.
set(ARM_CM_COMPILE_OPTIONS
  -O0
  -ffast-math
  -fno-finite-math-only
  -g0
)

# Build ARM CM source code to objects using clang without linking. When function
# returns the ARM_CM_OBJECT_FILES list is defined containing the object files.
glow_add_libjit(
  NAME "ARM_CM"
  SOURCE_FILES ${ARM_CM_SOURCE_FILES}
  COMPILE_OPTIONS ${ARM_CM_COMPILE_OPTIONS}
  CLANG_BIN clang
  OPTION COMPILE_ONLY
)

# Add object files to CPU list.
list(APPEND LIBJIT_CPU_OBJECT_FILES
  ${ARM_CM_OBJECT_FILES}
)

# Add source files and object files to CPU dependencies.
list(APPEND LIBJIT_CPU_DEPENDENCIES
  ${ARM_CM_SOURCE_FILES} ${ARM_CM_OBJECT_FILES}
)

#===============================================================================

# Add CPU LIBJIT.
glow_add_libjit(
  NAME "libjit_cpu"
  SOURCE_FILES ${LIBJIT_CPU_SOURCE_FILES}
  OBJECT_FILES ${LIBJIT_CPU_OBJECT_FILES}
  DEPENDS ${LIBJIT_CPU_DEPENDENCIES}
  COMPILE_OPTIONS ${LIBJIT_CPU_COMPILE_OPTIONS}
)

# Add native CPU LIBJIT library used for testing.
if (NOT MSVC)
  add_library(CPURuntimeNative
              ${LIBJIT_CPU_SOURCE_FILES})
endif(NOT MSVC)

add_library(CPUBackend
            ${libjit_cpu_INCLUDE_FILE}
            CPUBackend.cpp
            CPUDeviceManager.cpp
            CPUFactory.cpp
            CPUFunction.cpp
            CPULLVMIRGen.cpp
            Transforms.cpp)

target_link_libraries(CPUBackend
                      PUBLIC
                        Backend
                        Base
                        CodeGen
                        Flags
                        Graph
                        IR
                        IROptimizer
                        IROptimizerPipeline
                        GraphOptimizerPipeline
                        QuantizationBase
                        Runtime
                        LLVMIRCodeGen)

add_dependencies(CPUBackend libjit_cpu_TARGET)

set(linked_backends ${linked_backends} CPUBackend PARENT_SCOPE)
