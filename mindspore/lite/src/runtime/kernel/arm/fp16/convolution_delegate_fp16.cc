/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/runtime/kernel/arm/fp16/convolution_delegate_fp16.h"
#include <vector>
#include "src/runtime/kernel/arm/fp32/convolution_delegate_fp32.h"
#include "src/runtime/kernel/arm/fp16/convolution_fp16.h"
#include "src/runtime/kernel/arm/fp16/convolution_winograd_fp16.h"
#include "src/runtime/kernel/arm/fp16/convolution_1x1_fp16.h"
#include "src/runtime/kernel/arm/fp16/group_convolution_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2D;
using mindspore::schema::Format::Format_NHWC;

namespace mindspore::kernel {
void ConvolutionDelegateFP16CPUKernel::FreeCopiedData() {
  if ((fp16_weight_ != nullptr) && (need_free_ & WEIGHT_NEED_FREE)) {
    free(fp16_weight_);
    fp16_weight_ = nullptr;
  }
  if ((fp16_bias_ != nullptr) && (need_free_ & BIAS_NEED_FREE)) {
    free(fp16_bias_);
    fp16_bias_ = nullptr;
  }
}

int ConvolutionDelegateFP16CPUKernel::GetFp16WeightAndBias() {
  auto ret = GetFp16Weight();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get Fp16 Weight failed.";
    return RET_ERROR;
  }

  ret = GetFp16Bias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get Fp16 Bias failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDelegateFP16CPUKernel::GetFp16Weight() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  if (weight_tensor->data_type() == kNumberTypeFloat16 && InferShapeDone()) {
    // do not need malloc new memory to store origin data
    fp16_weight_ = reinterpret_cast<float16_t *>(weight_tensor->data_c());
    return RET_OK;
  } else {
    fp16_weight_ = CopyData(weight_tensor);
    if (fp16_weight_ == nullptr) {
      MS_LOG(ERROR) << "Generate fp16_weight failed.";
      return RET_ERROR;
    }
    need_free_ = need_free_ | WEIGHT_NEED_FREE;
    return RET_OK;
  }
  return RET_OK;
}

int ConvolutionDelegateFP16CPUKernel::GetFp16Bias() {
  if (in_tensors_.size() == 3) {
    // has bias situation
    auto bias_tensor = in_tensors_.at(kBiasIndex);
    if (bias_tensor->data_type() == kNumberTypeFloat16 && InferShapeDone()) {
      // do not need malloc new memory to store origin data
      fp16_bias_ = reinterpret_cast<float16_t *>(bias_tensor->data_c());
      return RET_OK;
    } else {
      fp16_bias_ = CopyData(bias_tensor);
      if (fp16_bias_ == nullptr) {
        MS_LOG(ERROR) << "Generate fp16_bias failed.";
        return RET_ERROR;
      }
      need_free_ = need_free_ | BIAS_NEED_FREE;
      return RET_OK;
    }
  }
  return RET_OK;
}

float16_t *ConvolutionDelegateFP16CPUKernel::CopyData(lite::Tensor *tensor) {
  auto data_type = tensor->data_type();
  MS_ASSERT(data_type == kNumberTypeFloat32 || data_type == kNumberTypeFloat16);
  auto fp16_data = reinterpret_cast<float16_t *>(malloc(tensor->ElementsNum() * sizeof(float16_t)));
  if (fp16_data == nullptr) {
    MS_LOG(ERROR) << "Malloc fp16_data failed.";
    return nullptr;
  }
  if (data_type == kNumberTypeFloat32) {
    float *origin_data = reinterpret_cast<float *>(tensor->data_c());
    for (size_t i = 0; i < tensor->ElementsNum(); ++i) {
      fp16_data[i] = (float16_t)origin_data[i];
    }
  } else {
    auto *origin_data = reinterpret_cast<float16_t *>(tensor->data_c());
    memcpy(fp16_data, origin_data, tensor->Size());
  }
  return fp16_data;
}

int ConvolutionDelegateFP16CPUKernel::Init() {
  auto ret = GetFp16WeightAndBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get fp16 weight and bias failed.";
    return ret;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionDelegateFP16CPUKernel::ReSize() {
  // Update shape info of input and output
  SetInputOutputShapeInfo(reinterpret_cast<ConvParameter *>(op_parameter_), in_tensors_.front(), out_tensors_.front(),
                          context_);
  if (fp16_conv_kernel_ == nullptr) {
    fp16_conv_kernel_ =
      CpuConvFp16KernelSelect(in_tensors_, out_tensors_, op_parameter_, context_, primitive_, fp16_weight_, fp16_bias_);
    if (fp16_conv_kernel_ == nullptr) {
      MS_LOG(ERROR) << "Selecting execute kernel failed for conv_kernel, got a nullptr.";
      return RET_ERROR;
    }
  }
  // copied weight and bias are not be used anymore,free them.
  FreeCopiedData();
  return fp16_conv_kernel_->ReSize();
}

ConvParameter *CreateNewConvParameterFp16(ConvParameter *parameter) {
  auto conv_parameter = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (conv_parameter == nullptr) {
    MS_LOG(ERROR) << "Malloc new conv parameter failed.";
    return nullptr;
  }
  memcpy(conv_parameter, parameter, sizeof(ConvParameter));
  return conv_parameter;
}

kernel::LiteKernel *CpuConvFp16KernelSelect(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                            const lite::InnerContext *ctx, const mindspore::lite::PrimitiveC *primitive,
                                            float16_t *fp16_weight, float16_t *fp16_bias) {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  bool use_winograd = false;
  int out_unit;
  CheckIfUseWinogradFp16(&use_winograd, &out_unit, conv_param);
  kernel::LiteKernel *kernel = nullptr;
  if (conv_param->kernel_h_ == 1 && conv_param->kernel_w_ == 1) {
    kernel = new (std::nothrow)
      kernel::Convolution1x1FP16CPUKernel(op_parameter, inputs, outputs, ctx, primitive, fp16_weight, fp16_bias);
  } else if (use_winograd) {
    kernel = new (std::nothrow) kernel::ConvolutionWinogradFP16CPUKernel(op_parameter, inputs, outputs, ctx, primitive,
                                                                         out_unit, fp16_weight, fp16_bias);
  } else {
    kernel = new (std::nothrow)
      kernel::ConvolutionFP16CPUKernel(op_parameter, inputs, outputs, ctx, primitive, fp16_weight, fp16_bias);
  }
  // Once kernel is selected, init func will invoke InitWeightAndBias
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "kernel init failed.";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

void FreeMemoryFp16(const std::vector<kernel::LiteKernel *> &group_convs, const std::vector<lite::Tensor *> &new_inputs,
                    const std::vector<lite::Tensor *> &new_outputs) {
  for (auto sub_conv : group_convs) {
    delete sub_conv;
  }
  for (auto in_tensor : new_inputs) {
    delete in_tensor;
  }
  for (auto out_tensor : new_outputs) {
    delete out_tensor;
  }
}

static lite::Tensor *CreateInputTensorFp16(TypeId data_type, const std::vector<int> &in_shape, bool infered_flag) {
  auto in_tensor = new (std::nothrow) lite::Tensor(data_type, in_shape, Format_NHWC, lite::Tensor::Category::VAR);
  if (in_tensor == nullptr) {
    MS_LOG(ERROR) << "new in_tensor failed.";
    return nullptr;
  }
  if (infered_flag) {
    auto ret = in_tensor->MallocData();
    if (ret != RET_OK) {
      delete in_tensor;
      MS_LOG(ERROR) << "in tensor malloc failed.";
      return nullptr;
    }
  }
  return in_tensor;
}

static lite::Tensor *CreateConstTensorFp16(lite::Tensor *tensor, const std::vector<int> &shape, const int index) {
  auto new_tensor =
    new (std::nothrow) lite::Tensor(tensor->data_type(), shape, Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  if (new_tensor == nullptr) {
    MS_LOG(ERROR) << "Create new_tensor failed.";
    return nullptr;
  }
  auto ret = new_tensor->MallocData();
  if (ret != RET_OK) {
    delete new_tensor;
    MS_LOG(ERROR) << "Malloc new_tensor failed.";
    return nullptr;
  }
  memcpy(new_tensor->data_c(), reinterpret_cast<char *>(tensor->data_c()) + index * new_tensor->Size(),
         new_tensor->Size());
  return new_tensor;
}

static lite::Tensor *CreateOutputTensorFp16(const std::vector<int> &out_shape,
                                            const std::vector<lite::Tensor *> &outputs, bool infered_flag, int index) {
  auto out_tensor = new (std::nothrow) lite::Tensor();
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "new tmp_out_tensor failed.";
    return nullptr;
  }
  out_tensor->set_data_type(mindspore::kNumberTypeFloat16);
  out_tensor->set_format(outputs.at(index)->format());
  if (infered_flag) {
    out_tensor->set_shape(out_shape);
    auto ret = out_tensor->MallocData();
    if (ret != RET_OK) {
      delete out_tensor;
      MS_LOG(ERROR) << "out_tensor malloc data failed.";
      return nullptr;
    }
  }
  return out_tensor;
}

kernel::LiteKernel *CreateDelegateConvFp16(const std::vector<lite::Tensor *> &inputs,
                                           const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                           const InnerContext *ctx, const mindspore::lite::PrimitiveC *primitive) {
  return new (std::nothrow) kernel::ConvolutionDelegateFP16CPUKernel(op_parameter, inputs, outputs, ctx, primitive);
}

kernel::LiteKernel *CpuGroupConvFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                  const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                                  const InnerContext *ctx,
                                                  const mindspore::lite::PrimitiveC *primitive) {
  bool infer_flag = (primitive != nullptr && primitive->infer_flag());
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  // update new shape info for each sub kernel
  int new_in_channel = inputs.at(kWeightIndex)->Channel();
  int new_out_channel = 0;
  if (conv_param->group_ == 0) {
    MS_LOG(ERROR) << "Divisor 'group' cannot be 0.";
    return nullptr;
  } else {
    new_out_channel = inputs.at(kWeightIndex)->Batch() / conv_param->group_;
  }

  std::vector<int> in_shape;
  std::vector<int> out_shape;
  if (infer_flag) {
    conv_param->input_channel_ = new_in_channel;
    conv_param->output_channel_ = new_out_channel;
    in_shape = {inputs.front()->Batch(), inputs.front()->Height(), inputs.front()->Width(), new_in_channel};
    out_shape = {inputs.front()->Batch(), outputs.front()->Height(), outputs.front()->Width(), new_out_channel};
  }
  std::vector<int> filter_shape = {new_out_channel, conv_param->kernel_h_, conv_param->kernel_w_, new_in_channel};
  std::vector<int> bias_shape = {new_out_channel};

  // new group conv op
  std::vector<kernel::LiteKernel *> group_convs;
  // create tensors for every sub conv kernel
  for (int i = 0; i < conv_param->group_; ++i) {
    std::vector<lite::Tensor *> new_inputs;
    std::vector<lite::Tensor *> new_outputs;
    auto new_conv_parameter = CreateNewConvParameterFp16(conv_param);
    if (new_conv_parameter == nullptr) {
      FreeMemoryFp16(group_convs, new_inputs, new_outputs);
      MS_LOG(ERROR) << "Get new conv parameter failed.";
      return nullptr;
    }
    // create new input for each group
    auto in_tensor = CreateInputTensorFp16(mindspore::kNumberTypeFloat16, in_shape, infer_flag);
    if (in_tensor == nullptr) {
      delete new_conv_parameter;
      FreeMemoryFp16(group_convs, new_inputs, new_outputs);
      MS_LOG(ERROR) << "create input tensor failed.";
      return nullptr;
    }
    new_inputs.emplace_back(in_tensor);

    // create new weight
    auto filter_tensor = CreateConstTensorFp16(inputs.at(kWeightIndex), filter_shape, i);
    if (filter_tensor == nullptr) {
      delete new_conv_parameter;
      FreeMemoryFp16(group_convs, new_inputs, new_outputs);
      MS_LOG(ERROR) << "create filter tensor failed.";
      return nullptr;
    }
    new_inputs.emplace_back(filter_tensor);

    // if has bias, create new bias
    if (inputs.size() == 3) {
      auto bias_tensor = CreateConstTensorFp16(inputs.at(kBiasIndex), bias_shape, i);
      if (bias_tensor == nullptr) {
        delete new_conv_parameter;
        FreeMemoryFp16(group_convs, new_inputs, new_outputs);
        MS_LOG(ERROR) << "create bias_tensor failed.";
        return nullptr;
      }
      new_inputs.emplace_back(bias_tensor);
    }

    // create new output tensors
    for (size_t j = 0; j < outputs.size(); ++j) {
      auto out_tensor = CreateOutputTensorFp16(out_shape, outputs, infer_flag, j);
      if (out_tensor == nullptr) {
        delete new_conv_parameter;
        FreeMemoryFp16(group_convs, new_inputs, new_outputs);
        MS_LOG(ERROR) << "new out_tensor failed.";
        return nullptr;
      }
      new_outputs.emplace_back(out_tensor);
    }
    group_convs.emplace_back(CreateDelegateConvFp16(
      new_inputs, new_outputs, reinterpret_cast<OpParameter *>(new_conv_parameter), ctx, primitive));
  }
  return new (std::nothrow)
    GroupConvolutionFP16CPUKernel(op_parameter, inputs, outputs, ctx, primitive, group_convs, conv_param->group_);
}

kernel::LiteKernel *CpuConvFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                             const InnerContext *ctx, const kernel::KernelKey &desc,
                                             const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2D);

  auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  kernel::LiteKernel *kernel = nullptr;
  if (conv_param->group_ == 1) {
    kernel = CreateDelegateConvFp16(inputs, outputs, opParameter, ctx, primitive);
  } else {
    kernel = CpuGroupConvFp16KernelCreator(inputs, outputs, opParameter, ctx, primitive);
  }

  if (kernel == nullptr) {
    MS_LOG(DEBUG) << "Create conv fp16 kernel failed.";
    free(opParameter);
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(INFO) << "Init fp16 kernel failed, name: " << opParameter->name_
                 << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Conv2D, CpuConvFp16KernelCreator)
}  // namespace mindspore::kernel
