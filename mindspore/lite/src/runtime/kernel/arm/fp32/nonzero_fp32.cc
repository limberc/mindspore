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
#include "src/runtime/kernel/arm/fp32/nonzero_fp32.h"
#include "include/errorcode.h"
#include "nnacl/op_base.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "src/tensor.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_NonZero;

namespace mindspore::kernel {
int NonZeroCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int NonZeroCPUKernel::ReSize() { return RET_OK; }
int NonZeroCPUKernel::Run() {
  auto in_tensor = in_tensors_.front();
  auto out_tensor = out_tensors_.front();
  auto input_data = reinterpret_cast<float *>(in_tensor->MutableData());
  auto output_data = reinterpret_cast<int *>(out_tensor->MutableData());
  auto input_dim_size = in_tensor->shape().size();
  if (out_tensor->shape().size() != 2) {
    MS_LOG(ERROR) << "out tensor shape size must be equal to 2!";
    return RET_ERROR;
  }
  auto non_zero_nums = out_tensor->shape()[1];
  int non_zero_count = 0;
  std::vector coordiate_values(in_tensor->shape().size(), 0);
  for (int i = 0; i < in_tensor->ElementsNum(); i += 1) {
    if (input_data[i] != 0) {
      for (size_t j = 0; j < input_dim_size; j++) {
        output_data[non_zero_count + j * non_zero_nums] = coordiate_values[j];
      }
      non_zero_count++;
    }
    for (int idx = input_dim_size - 1; idx >= 0; --idx) {
      if (coordiate_values[idx] != in_tensor->shape()[idx] - 1) {
        coordiate_values[idx] = coordiate_values[idx] + 1;
        break;
      }
      coordiate_values[idx] = 0;
    }
  }
  return RET_OK;
}

kernel::LiteKernel *CpuNonZeroFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                const mindspore::lite::PrimitiveC *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "Input context is nullptr!";
    free(opParameter);
    return nullptr;
  }
  if (ctx->thread_num_ == 0) {
    MS_LOG(ERROR) << "context thread num is 0!";
    free(opParameter);
    return nullptr;
  }
  auto *kernel = new (std::nothrow) NonZeroCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new NonZeroCPUKernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_NonZero, CpuNonZeroFp32KernelCreator)
}  // namespace mindspore::kernel
