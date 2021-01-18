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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPLIT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPLIT_CPU_KERNEL_H_
#include <vector>
#include <memory>
#include <thread>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class SplitCPUKernel : public CPUKernel {
 public:
  SplitCPUKernel() = default;
  ~SplitCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

  void InitInputOutputSize(const CNodePtr &kernel_node) override;

 private:
  void CheckParam(const CNodePtr &kernel_node);
  void Reshape();
  void LaunchSplit(const T *input, T **output, size_t size);
  int64_t axis_;
  int64_t output_num_;
  int64_t axis_step_;

  size_t input_size_;
  size_t dims_after_axis_;
  size_t dims_current_after_axis_;

  std::vector<std::vector<size_t>> output_shape_list_;
  std::vector<size_t> input_shape_;
  TypeId dtype_{kTypeUnknown};
};

MS_REG_CPU_KERNEL_T(
  Split, KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  SplitCPUKernel, float);
MS_REG_CPU_KERNEL_T(
  Split, KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  SplitCPUKernel, float16);
MS_REG_CPU_KERNEL_T(
  Split, KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  SplitCPUKernel, double);
MS_REG_CPU_KERNEL_T(Split,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                    SplitCPUKernel, int32_t);
MS_REG_CPU_KERNEL_T(Split,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                    SplitCPUKernel, uint32_t);
MS_REG_CPU_KERNEL_T(Split,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                    SplitCPUKernel, int64_t);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPLIT_CPU_KERNEL_H_
