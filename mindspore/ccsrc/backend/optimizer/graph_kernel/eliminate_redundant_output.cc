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
#include "backend/optimizer/graph_kernel/eliminate_redundant_output.h"

#include <memory>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>

#include "base/core_ops.h"
#include "ir/graph_utils.h"
#include "backend/optimizer/common/helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "debug/anf_ir_dump.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"

namespace mindspore {
namespace opt {
namespace {
inline size_t GetIndex(const AnfNodePtr &getitem_node) {
  MS_EXCEPTION_IF_NULL(getitem_node);
  if (!IsPrimitiveCNode(getitem_node, prim::kPrimTupleGetItem)) {
    MS_LOG(EXCEPTION) << "User of MakeTuple should be GetItem but got " << getitem_node->fullname_with_scope();
  }
  return LongToSize(GetValue<int64_t>(
    getitem_node->cast<CNodePtr>()->input(kInputNodeOutputIndexInTupleGetItem)->cast<ValueNodePtr>()->value()));
}

void SetIndex(const AnfNodePtr &getitem_node, size_t index) {
  auto getitem = getitem_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(getitem);
  auto idx_node = NewValueNode(MakeValue<int64_t>(SizeToLong(index)));
  auto abstract = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int64Imm>(index));
  idx_node->set_abstract(abstract);
  idx_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  getitem->set_input(kInputNodeOutputIndexInTupleGetItem, idx_node);
}

bool GetGraphKernelGetitemList(const FuncGraphManagerPtr &mng, const AnfNodePtr &node, AnfNodePtrList *getitem_list,
                               bool merge_repeated_getitem = false) {
  MS_EXCEPTION_IF_NULL(mng);
  MS_EXCEPTION_IF_NULL(getitem_list);
  auto func_graph = AnfAlgo::GetCNodeFuncGraphPtr(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto output = func_graph->output();
  if (!IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
    MS_LOG(EXCEPTION) << "The output should be a MakeTuple, but got " << output->fullname_with_scope();
  }
  auto output_num = output->cast<CNodePtr>()->size() - 1;
  getitem_list->clear();
  getitem_list->resize(output_num, nullptr);
  const auto &users = mng->node_users()[node];
  bool changed = false;
  AnfNodePtrList user_nodes;
  std::transform(users.begin(), users.end(), std::back_inserter(user_nodes),
                 [](const std::pair<AnfNodePtr, int> &user) { return user.first; });
  for (const auto &getitem : user_nodes) {
    MS_EXCEPTION_IF_NULL(getitem);
    auto idx = GetIndex(getitem);
    if (idx >= output_num) {
      MS_LOG(EXCEPTION) << "Index of GetItem is out of range of MakeTuple. getitem node: " << getitem->DebugString();
    }
    if (merge_repeated_getitem && (*getitem_list)[idx] != nullptr) {
      mng->Replace(getitem, (*getitem_list)[idx]);
      changed = true;
    } else {
      (*getitem_list)[idx] = getitem;
    }
  }
  return changed;
}

AnfNodePtrList FindGraphKernelsWithMultiOutput(const FuncGraphPtr &func_graph) {
  auto todos = TopoSort(func_graph->get_return());
  AnfNodePtrList result;
  std::copy_if(todos.begin(), todos.end(), std::back_inserter(result), [](const AnfNodePtr &node) {
    return AnfAlgo::IsGraphKernel(node) &&
           IsPrimitiveCNode(AnfAlgo::GetCNodeFuncGraphPtr(node)->output(), prim::kPrimMakeTuple);
  });
  return result;
}

/* Unify the repeated output in a func_graph.
 *   %1 = call @graph_kernel(p1, p2)
 *   %2 = tuple_getitem(%1, 0)
 *   %3 = tuple_getitem(%1, 1)
 *   graph_kernel:
 *      %1 = TensorAdd(p1, p2)
 *      %2 = Reshape(%1)
 *      return make_tuple(%2, %2)
 * -->
 *   %1 = call @graph_kernel(p1, p2)
 *   %2 = tuple_getitem(%1, 0)
 *   %3 = tuple_getitem(%1, 0)   // changed the index to 0.
 *   graph_kernel:
 *      %1 = TensorAdd(p1, p2)
 *      %2 = Reshape(%1)
 *      return make_tuple(%2, %2)
 */
class UnifyRepeatedOutput : public Pass {
 public:
  bool Run(const FuncGraphPtr &func_graph) {
    auto mng = func_graph->manager();
    MS_EXCEPTION_IF_NULL(mng);
    auto todos = FindGraphKernelsWithMultiOutput(func_graph);
    bool changed = false;
    for (auto node : todos) {
      if (CheckRepeatedOutput(AnfAlgo::GetCNodeFuncGraphPtr(node))) {
        changed = true;
        AnfNodePtrList getitem_list;
        GetGraphKernelGetitemList(mng, node, &getitem_list, false);
        if (getitem_list.size() != index_map_.size()) {
          MS_LOG(EXCEPTION) << "getitem_list.size (" << getitem_list.size() << ") should be equal to index_map.size ("
                            << index_map_.size() << ").";
        }
        for (size_t i = 0; i < index_map_.size(); ++i) {
          if (index_map_[i] != i && getitem_list[i] != nullptr) {
            SetIndex(getitem_list[i], index_map_[i]);
          }
        }
      }
    }
    return changed;
  }

 private:
  bool CheckRepeatedOutput(const FuncGraphPtr &sub_func_graph) {
    // the output should be a MakeTuple.
    auto maketuple = sub_func_graph->output()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(maketuple);
    AnfNodePtrList outputs(maketuple->inputs().begin() + 1, maketuple->inputs().end());
    index_map_.resize(outputs.size());
    bool found = false;
    for (size_t i = 0; i < outputs.size(); ++i) {
      index_map_[i] = std::find(outputs.begin(), outputs.begin() + i, outputs[i]) - outputs.begin();
      if (index_map_[i] != i) {
        found = true;
      }
    }
    return found;
  }
  std::vector<size_t> index_map_;
};

/* Unify the get_item nodes that have same index.
 *   %1 = call @graph_kernel(p1, p2)
 *   %2 = tuple_getitem(%1, 0)
 *   %3 = tuple_getitem(%1, 0)
 *   %4 = tuple_getitem(%1, 1)
 *   %5 = user_x(%2)
 *   %6 = user_y(%3)
 *   %7 = user_z(%4)
 *   --->
 *   %1 = call @graph_kernel(p1, p2)
 *   %2 = tuple_getitem(%1, 0) // unify the original %2 and %3
 *   %3 = tuple_getitem(%1, 1)
 *   %4 = user_x(%2)
 *   %5 = user_y(%2)
 *   %6 = user_z(%3)
 */
class UnifyRepeatedGetitem : public Pass {
 public:
  bool Run(const FuncGraphPtr &func_graph) {
    auto mng = func_graph->manager();
    MS_EXCEPTION_IF_NULL(mng);
    auto todos = FindGraphKernelsWithMultiOutput(func_graph);
    bool changed = false;
    for (auto node : todos) {
      AnfNodePtrList getitem_list;
      changed = GetGraphKernelGetitemList(mng, node, &getitem_list, true) || changed;
    }
    return changed;
  }
};

/* Merge the get_item nodes that have same index.
 *   subgraph graph_kernel(%para1, %para2)
 *     %1 = TensorAdd(%para1, %para2)
 *     %2 = Neg(%1)
 *     %3 = make_tuple(%1, %2)
 *     return (%3)
 *   %1 = call @graph_kernel(%p1, %p2)
 *   %2 = tuple_getitem(%1, 0)
 *   %3 = tuple_getitem(%1, 1)
 *   %4 = ControlDepend(%0, %2)
 *   %5 = other_user(%3)
 *   --->
 *   subgraph graph_kernel(%para1, %para2)
 *     %1 = TensorAdd(%para1, %para2)
 *     %2 = Neg(%1)
 *     %3 = make_tuple(%1, %2)
 *     return (%3)
 *   %1 = call @graph_kernel(%p1, %p2)
 *   %3 = tuple_getitem(%1, 1)
 *   %4 = ControlDepend(%0, %3)
 *   %5 = other_user(%3)
 *
 * Then the output 0 can be eliminate in the later pass.
 */
class EliminateGetitemForControlDepend : public Pass {
 public:
  bool Run(const FuncGraphPtr &func_graph) {
    auto todos = FindGraphKernelsWithMultiOutput(func_graph);
    auto mng = func_graph->manager();
    MS_EXCEPTION_IF_NULL(mng);
    bool changed = false;
    for (const auto &node : todos) {
      getitems_.clear();
      GetGraphKernelGetitemList(mng, node, &getitems_, false);
      if (getitems_.empty()) continue;
      indexes_.clear();
      GetIndexesToControlDepend(mng);
      FilterRedundantOutputs(node);
      if (indexes_.empty()) continue;
      size_t index = GetFinalIndex(node);
      changed = ReplaceGetitems(mng, index) || changed;
    }
    return changed;
  }

 private:
  AnfNodePtrList getitems_;      // Users of GraphKernel node with multiple outputs.
  std::vector<size_t> indexes_;  // Indexes of MakeTuple to be eliminated.

  bool ReplaceGetitems(const FuncGraphManagerPtr &mng, size_t index) {
    MS_EXCEPTION_IF_NULL(getitems_[index]);
    bool changed = false;
    for (auto i : indexes_) {
      if (i != index) {
        MS_EXCEPTION_IF_NULL(getitems_[i]);
        mng->Replace(getitems_[i], getitems_[index]);
        changed = true;
      }
    }
    return changed;
  }

  // Find the redundant output index.
  // the real output should have multiple users.
  void FilterRedundantOutputs(const AnfNodePtr &node) {
    auto func_graph = AnfAlgo::GetCNodeFuncGraphPtr(node);
    auto mng = func_graph->manager();
    if (mng == nullptr) {
      mng = Manage(func_graph, true);
      func_graph->set_manager(mng);
    }
    auto &users = mng->node_users();
    auto maketuple = func_graph->output()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(maketuple);
    std::vector<size_t> result;
    for (auto i : indexes_) {
      auto real_output = maketuple->input(i + 1);
      if (users[real_output].size() > 1) {
        result.push_back(i);
      }
    }
    indexes_ = std::move(result);
  }

  // Get the nodes that only have ControlDepend users.
  void GetIndexesToControlDepend(const FuncGraphManagerPtr &mng) {
    for (size_t i = 0; i < getitems_.size(); ++i) {
      const AnfNodePtr &getitem = getitems_[i];
      if (getitem == nullptr) {
        continue;
      }
      const auto &getitem_user = mng->node_users()[getitem];
      if (std::all_of(getitem_user.begin(), getitem_user.end(), [](const std::pair<AnfNodePtr, int> &user) {
            return IsPrimitiveCNode(user.first, prim::kPrimControlDepend);
          })) {
        indexes_.push_back(i);
      }
    }
  }

  size_t GetFinalIndex(const AnfNodePtr &node) {
    auto is_redundant_index = [this](size_t i) {
      return std::find(indexes_.begin(), indexes_.end(), i) != indexes_.end();
    };
    for (size_t i = 0; i < getitems_.size(); ++i) {
      if (getitems_[i] != nullptr && !is_redundant_index(i)) {
        return i;
      }
    }
    return indexes_[0];
  }
};
}  // namespace

// Remove the output without user or with virtual user (like ControlDepend)
bool EliminateRedundantOutput::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  bool changed = false;
  changed = std::make_shared<UnifyRepeatedGetitem>()->Run(func_graph) || changed;
  changed = std::make_shared<UnifyRepeatedOutput>()->Run(func_graph) || changed;
  changed = std::make_shared<UnifyRepeatedGetitem>()->Run(func_graph) || changed;
  changed = std::make_shared<EliminateGetitemForControlDepend>()->Run(func_graph) || changed;
  changed = Process(func_graph) || changed;
  return changed;
}

void EliminateRedundantOutput::UpdateGetitemIndex(const AnfNodePtr &getitem, size_t offset) {
  if (offset == 0) return;
  MS_EXCEPTION_IF_NULL(getitem);
  auto index = GetIndex(getitem);
  if (offset > index) {
    MS_LOG(EXCEPTION) << "The offset is greater than the original index of GetItem: " << getitem->DebugString();
  }
  index -= offset;
  SetIndex(getitem, index);
}

AnfNodePtr EliminateRedundantOutput::ReplaceMakeTuple(const AnfNodePtr &node, const AnfNodePtrList &getitems) {
  auto func_graph = AnfAlgo::GetCNodeFuncGraphPtr(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto old_maketuple = func_graph->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(old_maketuple);
  AnfNodePtrList new_maketuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  AbstractBasePtrList abstract_list;
  size_t offset = 0;
  for (size_t i = 0; i < getitems.size(); ++i) {
    if (getitems[i] == nullptr) {
      offset++;
    } else {
      new_maketuple_inputs.push_back(old_maketuple->input(i + 1));
      abstract_list.push_back(old_maketuple->input(i + 1)->abstract());
      UpdateGetitemIndex(getitems[i], offset);
    }
  }
  if (offset == 0) return nullptr;
  if (new_maketuple_inputs.size() == 1) {
    MS_LOG(EXCEPTION) << "Input of MakeTuple could not be empty";
  }
  if (new_maketuple_inputs.size() == 2) {
    func_graph->set_output(new_maketuple_inputs.back());
  } else {
    auto make_tuple = func_graph->NewCNode(new_maketuple_inputs);
    make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
    make_tuple->set_kernel_info(std::make_shared<device::KernelInfo>());
    func_graph->set_output(make_tuple);
  }

  auto old_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(old_cnode);
  AnfNodePtrList inputs(old_cnode->inputs().begin() + 1, old_cnode->inputs().end());
  AnfNodePtrList outputs;
  kernel::GetFuncGraphOutputNodes(func_graph, &outputs);
  auto graph_kernel_node = CreateNewFuseCNode(node->func_graph(), func_graph, inputs, outputs);
  SetNewKernelInfo(graph_kernel_node, func_graph, inputs, outputs, AnfAlgo::GetProcessor(node));
  return graph_kernel_node;
}

bool EliminateRedundantOutput::Process(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto todos = FindGraphKernelsWithMultiOutput(func_graph);
  bool changed = false;
  for (auto node : todos) {
    AnfNodePtrList getitems;
    GetGraphKernelGetitemList(mng, node, &getitems, false);
    auto new_node = ReplaceMakeTuple(node, getitems);
    if (new_node != nullptr) {
      if (!IsPrimitiveCNode(AnfAlgo::GetCNodeFuncGraphPtr(new_node)->output(), prim::kPrimMakeTuple)) {
        // only one output, remove the getitem.
        auto i = std::find_if(getitems.begin(), getitems.end(), [](const AnfNodePtr &node) { return node != nullptr; });
        if (i != getitems.end()) {
          mng->Replace(*i, new_node);
        }
      } else {
        mng->Replace(node, new_node);
      }
      changed = true;
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
