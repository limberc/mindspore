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

#include "src/runtime/kernel/arm/fp16/convolution_depthwise_fp16.h"
#include "src/runtime/kernel/arm/fp16/convolution_depthwise_slidewindow_fp16.h"
#include "nnacl/fp16/pack_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DepthwiseConv2D;

namespace mindspore::kernel {
ConvolutionDepthwiseFp16CPUKernel::~ConvolutionDepthwiseFp16CPUKernel() {
  if (packed_weight_ != nullptr) {
    free(packed_weight_);
    packed_weight_ = nullptr;
  }
}

int ConvolutionDepthwiseFp16CPUKernel::InitWeightBias() {
  // init weight: o, h, w, i; o == group, i == 1
  ConvolutionBaseFP16CPUKernel::GetExecuteFilter();

  auto weight_tensor = in_tensors_.at(kWeightIndex);
  int channel = weight_tensor->Batch();
  int pack_weight_size = channel * weight_tensor->Height() * weight_tensor->Width();

  packed_weight_ = reinterpret_cast<float16_t *>(malloc(pack_weight_size * sizeof(float16_t)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  PackNCHWToNHWCFp16(fp16_weight_, packed_weight_, 1, weight_tensor->Height() * weight_tensor->Width(),
                     weight_tensor->Batch());
  if (fp16_weight_ != nullptr) {
    free(fp16_weight_);
    fp16_weight_ = nullptr;
  }

  bias_data_ = reinterpret_cast<float16_t *>(malloc(channel * sizeof(float16_t)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, channel * sizeof(float16_t));
  auto bias_fp16 = reinterpret_cast<float16_t *>(bias_data_);
  if (in_tensors_.size() == kInputSize2) {
    auto bias_tensor = in_tensors_.at(kBiasIndex);
    auto ori_bias = reinterpret_cast<float *>(bias_tensor->MutableData());
    MS_ASSERT(ori_bias);
    for (int i = 0; i < bias_tensor->ElementsNum(); i++) {
      bias_fp16[i] = (float16_t)ori_bias[i];
    }
  }
  return RET_OK;
}

int ConvolutionDepthwiseFp16CPUKernel::Init() {
  auto ret = InitWeightBias();
  if (ret != 0) {
    MS_LOG(ERROR) << "Convolution depthwise fp16 InitWeightBias failed.";
    return RET_ERROR;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionDepthwiseFp16CPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }
  conv_param_->thread_num_ = MSMIN(thread_count_, conv_param_->output_h_);
  return RET_OK;
}

int ConvolutionDepthwiseFp16CPUKernel::Execute(int task_id) {
  ConvDwFp16(execute_output_, execute_input_, packed_weight_, reinterpret_cast<float16_t *>(bias_data_), conv_param_,
             task_id);
  return RET_OK;
}

static int ConvDwFp16Run(void *cdata, int task_id) {
  auto conv_dw_fp16 = reinterpret_cast<ConvolutionDepthwiseFp16CPUKernel *>(cdata);
  auto ret = conv_dw_fp16->Execute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionDepthwiseFp16Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseFp16CPUKernel::Run() {
  ConvolutionBaseFP16CPUKernel::GetExecuteTensor();

  auto ret = ParallelLaunch(this->context_->thread_pool_, ConvDwFp16Run, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvDwFp16Run error: error_code[" << ret << "]";
  }

  return ret;
}

kernel::LiteKernel *CpuConvDwFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                               const InnerContext *ctx, const kernel::KernelKey &desc,
                                               const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_DepthwiseConv2D);

  auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  kernel::LiteKernel *kernel;
  if (conv_param->input_channel_ < 32) {
    kernel =
      new (std::nothrow) kernel::ConvolutionDepthwiseSWFp16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  } else {
    kernel = new (std::nothrow) kernel::ConvolutionDepthwiseFp16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  }
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_DepthwiseConv2D, CpuConvDwFp16KernelCreator)
}  // namespace mindspore::kernel
