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

#include "Loader.h"
#include "glow/Graph/Graph.h"

#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Importer/ONNXModelLoader.h"

using namespace glow;

namespace {
llvm::cl::opt<std::string>
    preOpt("preproc", llvm::cl::desc("Input pre-processing function string."),
           llvm::cl::value_desc("func1(arg1,arg2)[,func2(arg1,arg2),...]"),
           llvm::cl::Optional);
}

#include <assert.h>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// Helper function to split strings.
static std::vector<std::string> splitString(std::string str, char delim = ',',
                                            std::string nopStartChars = "(",
                                            std::string nopStopChars = ")",
                                            std::string ignoreChars = " ") {
  assert(nopStartChars.size() == nopStopChars.size() &&
         "Start & end characters for split mismatch in size!");
  std::vector<std::string> partArray;
  std::string part = "";
  int openCount = 0;
  // Trim string and if empty then return empty array.
  while (ignoreChars.find(str[0]) != std::string::npos) {
    str = str.substr(1);
  }
  if (str.size() == 0) {
    return partArray;
  }
  // Split.
  auto strNew = str + delim;
  for (const char &ch : strNew) {
    if (nopStartChars.find(ch) != std::string::npos) {
      openCount++;
    }
    if (nopStopChars.find(ch) != std::string::npos) {
      openCount--;
    }
    if (openCount == 0) {
      // Ignore characters.
      if (ignoreChars.find(ch) != std::string::npos) {
        continue;
      }
      // Split action.
      if (ch == delim) {
        assert(part.size() != 0 && "Empty part while splitting string!");
        partArray.push_back(part);
        part = "";
      } else {
        part += ch;
      }
    } else {
      part += ch;
    }
  }
  assert(openCount == 0 &&
         "String incompatible with the start/stop characters!");
  return partArray;
}

// -----------------------------------------------------------------------------
//                               FUNCTION STRING
// -----------------------------------------------------------------------------
class FunctionString {

  // Function name.
  std::string name_;

  // Function arguments as strings.
  std::vector<std::string> args_;

  // Split argument into an array of arguments.
  static std::vector<std::string> splitArray(std::string arg);

public:
  // Ctors.
  FunctionString() = delete;
  FunctionString(std::string str);

  // Getter for function name.
  std::string getName() const { return name_; }

  // Getter for function arguments.
  std::vector<std::string> getArgs() const { return args_; }

  // Get number of function arguments.
  size_t getNumArgs() const { return args_.size(); }

  // Verify if function has argument.
  bool hasArg(int pos = 0) const;

  // Get an argument from the argument list.
  std::string getArg(int pos = 0) const;

  // Get a string argument.
  std::string getStrArg(int pos = 0) const;

  // Get a signed integer argument.
  int getArgInt(int pos = 0) const;

  // Get a float argument.
  float getArgFloat(int pos = 0) const;

  // Get a signed integer array argument.
  std::vector<int> getArgIntArray(int pos = 0) const;

  // Get a float array argument.
  std::vector<float> getArgFloatArray(int pos = 0) const;
};

FunctionString::FunctionString(std::string str) {
  auto braceStartPos = str.find('(');
  auto braceStopPos = str.find(')');
  if (braceStartPos == std::string::npos) {
    assert(braceStopPos == std::string::npos &&
           "Mismatch between opened/closed brackets!");
    // Function without brackets.
    name_ = str;
  } else {
    // Function with brackets.
    assert((braceStopPos == str.size() - 1) &&
           "Mismatch between opened/closed brackets!");
    assert(braceStartPos != 0 && "Function name empty!");
    name_ = str.substr(0, braceStartPos);
    auto argStr = str.substr(braceStartPos + 1, str.size() - name_.size() - 2);
    args_ = splitString(argStr, ',', "[", "]", " ");
  }
}

bool FunctionString::hasArg(int pos) const {
  return ((0 <= pos) && (pos < (int)args_.size()));
}

std::string FunctionString::getArg(int pos) const {
  assert(hasArg(pos) && "Invalid argument index!");
  return args_[pos];
}

std::string FunctionString::getStrArg(int pos) const { return getArg(pos); }

int FunctionString::getArgInt(int pos) const { return std::stoi(getArg(pos)); }

float FunctionString::getArgFloat(int pos) const {
  return std::stof(getArg(pos));
}

std::vector<std::string> FunctionString::splitArray(std::string arg) {
  assert(arg.size() > 2 && "Function array argument has invalid size!");
  assert(arg.front() == '[' && "Function array argument must start with '[' !");
  assert(arg.back() == ']' && "Function array argument must end with ']'.");
  return splitString(arg, ',', "", "", " []");
}

std::vector<int> FunctionString::getArgIntArray(int pos) const {
  std::vector<int> argIntArray;
  for (auto item : splitArray(getArg(pos))) {
    argIntArray.push_back(std::stoi(item));
  }
  return argIntArray;
}

std::vector<float> FunctionString::getArgFloatArray(int pos) const {
  std::vector<float> argIntArray;
  for (auto item : splitArray(getArg(pos))) {
    argIntArray.push_back(std::stof(item));
  }
  return argIntArray;
}

// -----------------------------------------------------------------------------
//                             FUNCTION STRING PARSER
// -----------------------------------------------------------------------------
class FunctionStringParser {

  // Array of function strings.
  std::vector<FunctionString> funcArray;

public:
  // Ctor.
  FunctionStringParser() = delete;
  FunctionStringParser(std::string str);

  // Get the function strings.
  std::vector<FunctionString> getFunctions() const { return funcArray; }

  // Get the number of functions.
  size_t getNumFunctions() const { return funcArray.size(); }
};

FunctionStringParser::FunctionStringParser(std::string str) {
  // Split functions.
  auto funcStrArray = splitString(str, ',', "([", ")]", " ");
  // Parse functions.
  for (auto funcStr : funcStrArray) {
    funcArray.push_back(FunctionString(funcStr));
  }
}

Node *createSimpleGraphFromString(Function *F, Node *input,
                                  std::string functionString) {

  // Parse the function string for this input.
  FunctionStringParser parser = FunctionStringParser(functionString);

  // -------------------------------------------------------------------------------
  std::cout << "------------- FUNCTIONS -------------------\n";
  std::cout << "Num functions = " << parser.getNumFunctions() << "\n";
  for (auto func : parser.getFunctions()) {
    auto funcName = func.getName();
    auto funcArgs = func.getArgs();
    std::cout << "Func = '" << funcName << "'\n";
    for (int idx = 0; idx < funcArgs.size(); idx++) {
      std::cout << "Arg[" << idx << "] = '" << funcArgs[idx] << "'\n";
    }
    std::cout << "\n";
  }
  // -------------------------------------------------------------------------------

  // Create simple linear graph.
  Node *node = input;
  for (const auto &func : parser.getFunctions()) {
    std::string funcName = func.getName();

    // Cast conversion.
    if (funcName == "CAST") {
      std::string type = func.getStrArg();
      if ((type == "FLOAT") || (type == "FLOAT32")) {
        node = F->createConvertTo("pre.CAST", node, ElemKind::FloatTy);
      } else if (type == "FLOAT16") {
        node = F->createConvertTo("pre.CAST", node, ElemKind::Float16Ty);
      } else if (type == "INT32") {
        node = F->createConvertTo("pre.CAST", node, ElemKind::Int32ITy);
      } else if (type == "INT64") {
        node = F->createConvertTo("pre.CAST", node, ElemKind::Int64ITy);
      } else {
        LOG(FATAL) << strFormat(
            "Graph string processor: for '%s' the type '%s' is not supported!",
            funcName.c_str(), type.c_str());
      }
      continue;
    }

    // NCHW to NHWC conversion.
    if (funcName == "NCHW2NHWC") {
      node = F->createTranspose("pre.NCHW2NHWC", node, NCHW2NHWC);
      continue;
    }

    // NHWC to NCHW conversion.
    if (funcName == "NHWC2NCHW") {
      node = F->createTranspose("pre.NHWC2NCHW", node, NHWC2NCHW);
      continue;
    }

    // Channel inversions.
    if ((funcName == "RGB2BGR") || (funcName == "BGR2RGB")) {
      int innerDim = (int)node->getType(0)->dims()[0];
      CHECK(innerDim == 3) << strFormat(
          "Graph string processor: for '%s' the tensor inner-most dimension "
          "must be 3 (encountered value is %d)!",
          funcName.c_str(), innerDim);
      // TODO: Implement Flip node.
      LOG(FATAL) << "Not supported yet ...";
      continue;
    }

    // RGB to YUV conversion.
    if (funcName == "RGB2YUV") {
      int innerDim = (int)node->getType(0)->dims()[0];
      CHECK(innerDim == 3) << strFormat(
          "Graph string processor: for '%s' the tensor inner-most dimension "
          "must be 3 (encountered value is %d)!",
          funcName.c_str(), innerDim);
      // TODO: Implement conversion node.
      LOG(FATAL) << "Not supported yet ...";
      continue;
    }

    // YUV to RGB conversion.
    if (funcName == "YUV2RGB") {
      int innerDim = (int)node->getType(0)->dims()[0];
      CHECK(innerDim == 3) << strFormat(
          "Graph string processor: for '%s' the tensor inner-most dimension "
          "must be 3 (encountered value is %d)!",
          funcName.c_str(), innerDim);
      // TODO: Implement conversion node.
      LOG(FATAL) << "Not supported yet ...";
      continue;
    }

    // TODOs:
    // RGB2YUV, YUV2RGB
    // RGB2BGR, BGR2RGB -> FlipNode
    // NORM(min,max)
    // QUANT/DEQUANT operation??

    // Function not supported.
    LOG(FATAL) << strFormat(
        "Graph string processor: operator '%s' is not supported!",
        funcName.c_str());
  }
  CHECK(node->getNumResults() == 1)
      << "Graph string processor: last node must have only one output!";
  return node;
}

// ======================================================================================================
// ======================================================================================================
// ======================================================================================================
// ======================================================================================================

int main(int argc, char **argv) {

  // Verify/initialize command line parameters, and then loader initializes
  // the ExecutionEngine and Function.
  parseCommandLine(argc, argv);

  // Initialize loader.
  Loader loader;

  // Emit bundle flag should be true.
  CHECK(emittingBundle())
      << "Bundle output directory not provided. Use the -emit-bundle option!";

  // ======================================================================================================
  // ======================================================================================================
  // ======================================================================================================
  // ======================================================================================================
  // Get model input names and types.
  std::vector<std::string> inputNames;
  std::vector<Type> inputTypes;
  Loader::getModelInputs(inputNames, inputTypes);
  std::vector<const char *> inputNameRefs;
  std::vector<TypeRef> inputTypeRefs;
  for (size_t idx = 0, e = inputNames.size(); idx < e; idx++) {
    inputNameRefs.push_back(inputNames[idx].c_str());
    inputTypeRefs.push_back(&inputTypes[idx]);
  }

  // -------------------------------------------------------------------------
  // Notes:
  // 1. When pre-processing is required, the input tensor size is mandatory
  //    and it gives the tensor size of the effective model (with preprocessing
  //    subgraph added). If the pre-precessing string follows the tensor size
  //    in "-model-input" option then this is self-implied.
  // -------------------------------------------------------------------------
  // TODO: The subgraphs should be part of the '-model-input' option. For now
  //       set it here through another option.
  std::vector<std::string> inputFunctionStrings = {preOpt};

  // Loader function.
  Function *F = loader.getFunction();

  // Create input graphs (subgraphs).
  std::vector<Node *> inputGraphs(inputNames.size(), nullptr);
  for (size_t idx = 0, e = inputNames.size(); idx < e; idx++) {

    // Skip input graph creation if empty graph description.
    if (inputFunctionStrings[idx].empty()) {
      continue;
    }

    // Create new placeholder using the given name and type.
    Placeholder *newPlaceholder = F->getParent()->createPlaceholder(
        inputTypeRefs[idx], inputNameRefs[idx],
        /* isTrainable */ false);

    // Create input graph.
    inputGraphs[idx] = createSimpleGraphFromString(F, newPlaceholder,
                                                   inputFunctionStrings[idx]);

    // Update the type used to instatiate the model for this input using the
    // type of this graph output.
    inputTypeRefs[idx] = inputGraphs[idx]->getType(0);
  }

  // Create the model based on the input model format.
  std::unique_ptr<ProtobufLoader> protoLoader;
  if (!loader.getCaffe2NetDescFilename().empty()) {
    // For Caffe2 format the input placeholder names/types must be provided
    // explicitly. Get model input names and types.
    protoLoader.reset(new Caffe2ModelLoader(
        loader.getCaffe2NetDescFilename(), loader.getCaffe2NetWeightFilename(),
        inputNameRefs, inputTypeRefs, *loader.getFunction()));
  } else {
    // For ONNX format the input placeholders names/types are
    // derived automatically.
    protoLoader.reset(new ONNXModelLoader(loader.getOnnxModelFilename(),
                                          inputNameRefs, inputTypeRefs,
                                          *loader.getFunction()));
  }

  // Link the model graph to the input graphs.
  for (size_t idx = 0, e = inputNames.size(); idx < e; idx++) {

    // Skip this section if no subgraph was created for this input.
    if (!inputGraphs[idx]) {
      continue;
    }

    // Get old placeholder using the original name used for registration.
    Placeholder *oldPlaceholder =
        EXIT_ON_ERR(protoLoader->getInputByName(inputNames[idx]));

    // Replace old placeholder with the input graph.
    oldPlaceholder->getOutput().replaceAllUsesOfWith(inputGraphs[idx]);

    // Delete old placeholder.
    auto &vars = F->getParent()->getPlaceholders();
    F->getParent()->erasePlaceholder(
        std::find(vars.begin(), vars.end(), oldPlaceholder));
  }

  // ======================================================================================================
  // ======================================================================================================
  // ======================================================================================================
  // ======================================================================================================

  // Compile the model and generate the bundle.
  CompilationContext ctx;
  loader.compile(ctx);

  return 0;
}
