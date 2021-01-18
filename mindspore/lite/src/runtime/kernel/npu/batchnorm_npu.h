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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_BATCHNORM_NPU_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_BATCHNORM_NPU_H_

#include <vector>
#include "include/graph/op/all_ops.h"
#include "include/graph/compatible/all_ops.h"
#include "src/runtime/kernel/npu/npu_kernel.h"
#include "nnacl/batchnorm_parameter.h"

namespace mindspore::kernel {
class BatchnormNPUKernel : public NPUKernel {
 public:
  BatchnormNPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                     const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                     const mindspore::lite::PrimitiveC *primitive)
      : NPUKernel(parameter, inputs, outputs, ctx, primitive) {
    batchnorm_param_ = reinterpret_cast<BatchNormParameter *>(parameter);
  }
  ~BatchnormNPUKernel() override;

  int IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                OpParameter *opParameter) override;
  int SetNPUInputs(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                   const std::vector<ge::Operator *> &npu_inputs) override;
  ge::Operator *GetNPUOp() override;

 private:
  ge::op::BatchNormExt2 *batchnorm_ = nullptr;
  BatchNormParameter *batchnorm_param_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_BATCHNORM_NPU_H_
