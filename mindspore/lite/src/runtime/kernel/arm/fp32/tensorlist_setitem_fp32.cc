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
#include "include/errorcode.h"
#include "include/ms_tensor.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32/tensorlist_setitem_fp32.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TensorListSetItem;

namespace mindspore::kernel {

int TensorListSetItemCPUKernel::Init() { return RET_OK; }

int TensorListSetItemCPUKernel::Run() {
  input0_ = reinterpret_cast<lite::TensorList *>(in_tensors_[0]);
  if (dtype_ != input0_->tensors_data_type()) {
    MS_LOG(ERROR) << "op dtype:" << dtype_ << " is not equal in_tensors[0] dtype:" << input0_->data_type();
    return RET_ERROR;
  }
  int dim0 = input0_->ElementsNum() - 1;
  if (in_tensors_[1]->data_type() != kNumberTypeInt && in_tensors_[1]->data_type() != kNumberTypeInt32) {
    MS_LOG(ERROR) << "in_tensors_[1]->data_type():" << in_tensors_[1]->data_type() << " must be int";
    return RET_ERROR;
  }
  if (in_tensors_[1]->ElementsNum() != 1) {
    MS_LOG(ERROR) << "in_tensors_[1]->ElementsNum():" << in_tensors_[1]->ElementsNum() << " must be equal to 1!";
    return RET_ERROR;
  }
  index_ = reinterpret_cast<int *>(in_tensors_[1]->data_c())[0];
  if (index_ < 0 || index_ > dim0) {
    MS_LOG(ERROR) << "index tensor:[" << index_ << "] must be in [0, " << dim0 << "]!";
    return RET_ERROR;
  }
  input2_ = in_tensors_[2];
  MS_ASSERT(input2_ != nullptr);
  if (!input0_->IsCompatibleShape(input2_->shape())) {
    return RET_ERROR;
  }
  output0_ = reinterpret_cast<lite::TensorList *>(out_tensors_[0]);
  MS_ASSERT(output0_ != nullptr);
  // copy each tensor in tensors_
  for (int i = 0; i < output0_->ElementsNum(); ++i) {
    if (i == index_) {
      auto dst = output0_->GetTensor(i);
      if (dst == nullptr) {
        dst = lite::Tensor::CopyTensor(*input2_, true);
        auto &tensors = output0_->tensors();
        tensors.emplace_back(dst);
      } else {
        dst->set_data_type(input2_->data_type());
        dst->set_shape(input2_->shape());
        dst->set_format(input2_->format());
        dst->set_category(input2_->category());
        dst->set_root_tensor(input2_->root_tensor());
        dst->set_tensor_name(input2_->tensor_name());
        dst->set_quant_clusters(input2_->quant_clusters());
        auto ret = lite::Tensor::CopyTensorData(*input2_, dst);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "CopyTensorData[" << i << "] is failed!";
          return RET_ERROR;
        }
      }
    } else {
      auto src = input0_->GetTensor(i);
      auto dst = output0_->GetTensor(i);
      MS_ASSERT(src != nullptr);
      // merge move data will delete tensors
      if (dst == nullptr) {
        dst = lite::Tensor::CopyTensor(*src, src->data_c() != nullptr);
        auto &tensors = output0_->tensors();
        tensors.emplace_back(dst);
        continue;
      }

      if (src->data_type() != kTypeUnknown) {
        if (src->Size() != dst->Size()) {
          MS_LOG(ERROR) << "src->Size():" << src->Size() << " must be equal to dst->Size():" << dst->Size();
          return RET_ERROR;
        }
        auto ret = lite::Tensor::CopyTensorData(*src, dst);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "CopyTensorData[" << i << "] is failed!";
          return RET_ERROR;
        }
      }
    }
  }
  return RET_OK;
}

int TensorListSetItemCPUKernel::ReSize() { return RET_OK; }

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TensorListSetItem, LiteKernelCreator<TensorListSetItemCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_TensorListSetItem, LiteKernelCreator<TensorListSetItemCPUKernel>)
}  // namespace mindspore::kernel
