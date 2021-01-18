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

#include "backend/kernel_compiler/cpu/pad_and_shift_cpu_kernel.h"
#include <string>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void PadAndShiftCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  node_ = kernel_node;
  input_x_dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  type_size_ = GetTypeByte(TypeIdToType(input_x_dtype_));
  auto indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  batch_size_ = 1;
  for (size_t i = 0; i < indices_shape.size(); ++i) {
    batch_size_ *= indices_shape[i];
  }
  MS_LOG(INFO) << "PadAndShift batch_size:" << batch_size_;
  auto cum_sum_arr_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  if (cum_sum_arr_shape.size() != 1) {
    MS_LOG(ERROR) << "The shape of cum_sum_arr must be 1.";
  }
  cum_sum_size_ = cum_sum_arr_shape[0];
}

bool PadAndShiftCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> & /*workspace*/,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  if (input_x_dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else if (input_x_dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else {
    MS_LOG(ERROR) << "Dtype of input_x only support int32, int64";
    return false;
  }
  return true;
}

template <typename T>
void PadAndShiftCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  T *input_x = reinterpret_cast<T *>(inputs[0]->addr);
  T *cum_sum_arr = reinterpret_cast<T *>(inputs[1]->addr);
  T shift_idx = *reinterpret_cast<T *>(inputs[2]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);

  if (shift_idx >= static_cast<T>(cum_sum_size_)) {
    MS_LOG(EXCEPTION) << "Shift index must small than cumsum size.";
  }
  size_t output_size = cum_sum_arr[cum_sum_size_ - 1];
  T shift_size = cum_sum_arr[shift_idx];
  T valid_size = cum_sum_arr[shift_idx + 1] - shift_size;
  int ret = memset_s(output, outputs[0]->size, -1, type_size_ * output_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memset_s error, errorno" << ret;
  }
  ret = memcpy_s(output + shift_size, valid_size * type_size_, input_x, valid_size * type_size_);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno" << ret;
  }
  std::vector<size_t> out_shape;
  out_shape.emplace_back(output_size);
  std::vector<TypeId> dtypes;
  auto output_nums = AnfAlgo::GetOutputTensorNum(node_);
  for (size_t i = 0; i < output_nums; i++) {
    dtypes.push_back(AnfAlgo::GetOutputInferDataType(node_, i));
  }
  AnfAlgo::SetOutputInferTypeAndShape(dtypes, {out_shape}, node_.get());
}
}  // namespace kernel
}  // namespace mindspore
