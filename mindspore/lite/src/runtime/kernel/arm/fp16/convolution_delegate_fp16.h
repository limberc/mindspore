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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_DELEGATE_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_DELEGATE_FP16_H_

#include <arm_neon.h>
#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/op_base.h"

#define WEIGHT_NEED_FREE 0b01
#define BIAS_NEED_FREE 0b10

namespace mindspore::kernel {
class ConvolutionDelegateFP16CPUKernel : public LiteKernel {
 public:
  ConvolutionDelegateFP16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                                   const mindspore::lite::PrimitiveC *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive) {}
  ~ConvolutionDelegateFP16CPUKernel() override {
    FreeCopiedData();
    if (fp16_conv_kernel_ != nullptr) {
      op_parameter_ = nullptr;  // set op_parameter of delegate to nullptr, avoiding double free
      delete fp16_conv_kernel_;
      fp16_conv_kernel_ = nullptr;
    }
  }
  int GetFp16WeightAndBias();
  int GetFp16Weight();
  int GetFp16Bias();
  float16_t *CopyData(lite::Tensor *tensor);
  void FreeCopiedData();
  int Init() override;
  int ReSize() override;
  int Run() override { return fp16_conv_kernel_->Run(); }

 private:
  uint8_t need_free_ = 0b00;
  kernel::LiteKernel *fp16_conv_kernel_ = nullptr;
  float16_t *fp16_weight_ = nullptr;
  float16_t *fp16_bias_ = nullptr;
};

kernel::LiteKernel *CpuConvFp16KernelSelect(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                            const lite::InnerContext *ctx, const mindspore::lite::PrimitiveC *primitive,
                                            float16_t *fp16_weight, float16_t *fp16_bias);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_DELEGATE_FP16_H_
