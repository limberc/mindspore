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
#include "src/runtime/kernel/arm/fp16/pooling_fp16.h"
#include <vector>
#include "nnacl/fp16/pooling_fp16.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"
#include "nnacl/op_base.h"
#include "src/runtime/kernel/arm/fp16/common_fp16.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Pooling;

namespace mindspore::kernel {
int PoolingFp16CPUKernel::Init() {
  auto ret = PoolingBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PoolingBase Init failed.";
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int PoolingFp16CPUKernel::ReSize() {
  auto ret = PoolingBaseCPUKernel::ReSize();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PoolingBase ReSize fai1!ret: " << ret;
    return ret;
  }
  return RET_OK;
}

int PoolingFp16CPUKernel::RunImpl(int task_id) {
  float16_t minf = -FLT_MAX;
  float16_t maxf = FLT_MAX;
  if (pooling_param_->act_type_ == ActType_Relu) {
    minf = 0.f;
  } else if (pooling_param_->act_type_ == ActType_Relu6) {
    minf = 0.f;
    maxf = 6.f;
  }
  if (pooling_param_->pool_mode_ == PoolMode_MaxPool) {
    MaxPoolingFp16(fp16_input_, fp16_output_, pooling_param_, task_id, minf, maxf);
  } else {
    auto ret = AvgPoolingFp16(fp16_input_, fp16_output_, pooling_param_, task_id, minf, maxf);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "AvgPooling run failed.";
      return ret;
    }
  }
  return RET_OK;
}

static int PoolingFp16Impl(void *cdata, int task_id) {
  auto pooling = reinterpret_cast<PoolingFp16CPUKernel *>(cdata);
  auto error_code = pooling->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Pooling Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PoolingFp16CPUKernel::Run() {
  auto input_tensor = in_tensors_.at(0);
  auto output_tensor = out_tensors_.at(0);

  fp16_input_ = reinterpret_cast<float16_t *>(input_tensor->data_c());
  fp16_output_ = reinterpret_cast<float16_t *>(output_tensor->data_c());

  int error_code = ParallelLaunch(this->context_->thread_pool_, PoolingFp16Impl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "pooling error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Pooling, LiteKernelCreator<PoolingFp16CPUKernel>)
}  // namespace mindspore::kernel
