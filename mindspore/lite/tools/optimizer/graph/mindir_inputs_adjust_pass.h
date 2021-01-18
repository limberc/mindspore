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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_MINDIR_INPUTS_ADJUST_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_MINDIR_INPUTS_ADJUST_PASS_H_

#include <string>
#include <vector>
#include "backend/optimizer/common/pass.h"
#include "tools/converter/converter_flags.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/param_value_lite.h"

namespace mindspore::opt {
class MindirInputAdjustOpPass : public Pass {
 public:
  MindirInputAdjustOpPass() : Pass("mindir_inputs_adjust_pass") {}
  ~MindirInputAdjustOpPass() override = default;
  bool CheckCNodeIsArgMinMax(const CNodePtr &cnode);
  int AdjustArgMinMaxInputs(std::vector<AnfNodePtr> *inputs, bool index_or_value);
  int CopyPrimitiveCForArgMinMax(std::vector<AnfNodePtr> *inputs);
  int BuildCNodeForArgMinMax(const FuncGraphPtr &graph, const CNodePtr &tuple_get_item, const CNodePtr &argmin_max);
  int AdjustArgMinMax(const FuncGraphPtr &graph, const CNodePtr &tuple_get_item, const CNodePtr &argmin_max);
  int AdjustTupleGetItemWithArgMinMax(const FuncGraphPtr &graph, const CNodePtr &cnode);
  bool Run(const FuncGraphPtr &graph) override;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_MINDIR_INPUTS_ADJUST_PASS_H_
