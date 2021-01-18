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

#include <functional>
#include <vector>
#include "include/errorcode.h"
#include "ir/dtype/type_id.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32/tensorlist_stack_fp32.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TensorListStack;

namespace mindspore::kernel {

int TensorListStackCPUKernel::CheckParam() {
  if (input0_->tensors_data_type() != dtype_) {
    MS_LOG(ERROR) << "in_tensors_[0].tensors_data_type:[" << input0_->tensors_data_type() << "] must be equal "
                  << "param.data_type:[" << dtype_ << "]";
    return RET_ERROR;
  }
  if (num_element_ != -1 && input0_->ElementsNum() != num_element_) {
    MS_LOG(ERROR) << "in_tensors_[0].ElementsNum():[" << input0_->ElementsNum() << "] must be equal "
                  << "param.elements_num:[" << num_element_ << "]";
    return RET_ERROR;
  }
  num_element_ = input0_->ElementsNum();
  if (output0_->shape().size() < 1) {
    MS_LOG(ERROR) << "out_tensors_[0].shape().size():" << output0_->shape().size()
                  << " must be greater than or equal to 1!";
    return RET_ERROR;
  }
  int dim0 = output0_->shape()[0];
  if (dim0 != num_element_) {
    MS_LOG(ERROR) << "out_tensors_[0].shape()[0] must be:" << num_element_ << ", but now is:" << dim0;
    return RET_ERROR;
  }
  return RET_OK;
}

int TensorListStackCPUKernel::Init() {
  input0_ = reinterpret_cast<lite::TensorList *>(in_tensors_[0]);
  MS_ASSERT(input0_ != nullptr);
  output0_ = out_tensors_[0];
  MS_ASSERT(output0_ != nullptr);
  return RET_OK;
}

bool TensorListStackCPUKernel::IsFullyDefined(const std::vector<int> &shape) const {
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0) {
      return false;
    }
  }
  return true;
}

int TensorListStackCPUKernel::MergeElementShape() {
  MS_ASSERT(in_tensors_[1]);
  if (in_tensors_[1]->data_type() != kNumberTypeInt && in_tensors_[1]->data_type() != kNumberTypeInt32) {
    MS_LOG(ERROR) << "in_tensors_[1]->data_type():" << in_tensors_[1]->data_type() << " must be int";
    return RET_ERROR;
  }
  auto ele_shape_data = reinterpret_cast<int *>(in_tensors_[1]->data_c());
  output_shape_.clear();
  for (int i = 0; i < in_tensors_[1]->ElementsNum(); ++i) {
    output_shape_.push_back(ele_shape_data[i]);
  }
  auto status = MergeSubShape(input0_->element_shape());
  if (status == RET_ERROR) {
    MS_LOG(ERROR) << "Merge element_shape is error!";
    return RET_ERROR;
  }

  if (!IsFullyDefined(output_shape_)) {
    MS_LOG(ERROR) << "output_shape_ Is Not FullyDefined!";
    return RET_ERROR;
  }
  if (!IsFullyDefined(input0_->element_shape())) {
    for (int i = 0; i < input0_->ElementsNum(); ++i) {  // get tensorlist every tensor
      auto tensor_ele = input0_->GetTensor(i);
      MS_ASSERT(tensor_ele != nullptr);
      if (tensor_ele->data_type() != kTypeUnknown) {
        status = MergeSubShape(tensor_ele->shape());
        if (status == RET_ERROR) {
          MS_LOG(ERROR) << "Merge tensors_[" << i << "] is error!";
          return RET_ERROR;
        }
      }
    }
  }
  TypeUnknownSize = std::accumulate(output_shape_.begin(), output_shape_.end(), 1LL, std::multiplies<int>());
  return RET_OK;
}

int TensorListStackCPUKernel::MergeSubShape(const std::vector<int> &shape) {
  size_t dim0 = shape.size();
  size_t dim1 = output_shape_.size();
  if (dim1 != dim0) {
    MS_LOG(ERROR) << "shape.size():" << dim1 << " must be equal output_shape_.size():" << dim0;
    return RET_ERROR;
  }
  for (size_t i = 0; i < dim0; ++i) {
    int dim0_size = shape[i];
    int dim1_size = output_shape_[i];
    if (dim0_size >= 0 && dim1_size >= 0 && dim0_size != dim1_size) {
      MS_LOG(ERROR) << "shape[" << i << "]:" << dim0_size << " is incompatible with output_shape_[" << i
                    << "]:" << dim1_size;
      return RET_ERROR;
    }
    output_shape_[i] = dim1_size >= 0 ? dim1_size : dim0_size;
  }
  return RET_OK;
}

int TensorListStackCPUKernel::Run() {
  if (CheckParam() != RET_OK) {
    MS_LOG(ERROR) << "CheckParam failed!";
    return RET_ERROR;
  }
  if (output0_->ElementsNum() == 0) {
    return RET_OK;
  }
  auto ret = MergeElementShape();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MergeElementShape failed!";
    return RET_ERROR;
  }
  size_t in_ele_num = num_element_ * TypeUnknownSize;
  size_t out_ele_num = output0_->ElementsNum();
  if (in_ele_num != out_ele_num) {
    MS_LOG(ERROR) << "out_tensors_[0]->ElementsNum():" << out_ele_num << "must be equal to in_ele_num:" << in_ele_num;
    return RET_ERROR;
  }
  auto out_ptr = reinterpret_cast<float *>(output0_->MutableData());
  for (int i = 0; i < num_element_; ++i) {
    auto in_ptr = input0_->GetTensor(i);
    MS_ASSERT(in_ptr != nullptr);
    if (in_ptr->data_type() != kTypeUnknown) {
      int in_size = in_ptr->ElementsNum();
      memcpy(out_ptr, in_ptr->data_c(), in_size * sizeof(float));
      out_ptr += in_size;
    } else {
      memset(out_ptr, 0, TypeUnknownSize * sizeof(float));
      out_ptr += TypeUnknownSize;
    }
  }
  return RET_OK;
}

int TensorListStackCPUKernel::ReSize() { return RET_OK; }

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TensorListStack, LiteKernelCreator<TensorListStackCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_TensorListStack, LiteKernelCreator<TensorListStackCPUKernel>)
}  // namespace mindspore::kernel
