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
#include "tools/optimizer/graph/while_pass.h"
#include <vector>
#include <memory>
#include <algorithm>
#include "mindspore/lite/include/errorcode.h"
#include "mindspore/lite/src/ops/primitive_c.h"
#include "tools/anf_importer/import_from_meta_graphT.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/ops/primitive_c.h"
#include "schema/inner/model_generated.h"
#include "src/tensor.h"
#include "src/common/log_adapter.h"
#include "src/ops/switch.h"
#include "src/ops/partial.h"

namespace mindspore::opt {

ValueNodePtr WhilePass::GetSwitchAnfPrim() {
  auto switch_primitiveT = new (std::nothrow) schema::PrimitiveT;
  if (switch_primitiveT == nullptr) {
    MS_LOG(ERROR) << "new switch_primitiveT failed";
    return nullptr;
  }
  switch_primitiveT->value.type = schema::PrimitiveType_Switch;
  switch_primitiveT->value.value = new (std::nothrow) schema::SwitchT;
  if (switch_primitiveT->value.value == nullptr) {
    MS_LOG(ERROR) << "new MakeTupleT failed";
    delete (switch_primitiveT);
    return nullptr;
  }

  auto partial_prim = std::make_shared<lite::Partial>(switch_primitiveT);
  ValueNodePtr partial_anf_prim = NewValueNode(partial_prim);
  return partial_anf_prim;
}

void WhilePass::ReplaceInput(const std::vector<AnfNodePtr> &node_list, AnfNodePtr new_input_cnode,
                             std::string para_name) {
  for (auto &node : node_list) {
    if (utils::isa<CNodePtr>(node)) {
      auto cnode = utils::cast<CNodePtr>(node);
      for (size_t k = 0; k < cnode->inputs().size(); k++) {
        if (!utils::isa<ParameterPtr>(cnode->input(k))) {
          continue;
        }
        auto para_input = utils::cast<ParameterPtr>(cnode->input(k));
        if (para_input->name() == para_name) {
          cnode->set_input(k, new_input_cnode);
        }
      }
    }
  }
}

bool WhilePass::Run(const FuncGraphPtr &graph) {
  auto node_list = TopoSort(graph->get_return());
  static int count = 0;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (opt::GetCNodeType(node) != schema::PrimitiveType_While) {
      continue;
    }
    auto while_cnode = node->cast<CNodePtr>();
    MS_ASSERT(while_cnode != nullptr);
    if (while_cnode->inputs().size() < kWhileMinInputSize) {
      MS_LOG(ERROR) << "while input is not right.";
      return false;
    }

    // the order is fixed.
    auto cond_vnode = while_cnode->input(kWhileCondIndex);
    auto body_vnode = while_cnode->input(kWhileBodyIndex);

    // body_vnode->cast<ValueNodePtr>()->set_value()
    auto cond_fg = GetValueNode<std::shared_ptr<FuncGraph>>(cond_vnode);
    auto body_fg = GetValueNode<std::shared_ptr<FuncGraph>>(body_vnode);

    if (cond_fg == nullptr || body_fg == nullptr) {
      MS_LOG(ERROR) << "Get value as func_graph failed.";
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_FAILED);
      return false;
    }

    // create cond partial cnode
    std::vector<AnfNodePtr> cond_partial_op_inputs{cond_vnode};

    // create body partial cnode
    std::vector<AnfNodePtr> body_partial_op_inputs{body_vnode};

    // add while op input to cond_cnode and body_cnode
    cond_partial_op_inputs.insert(cond_partial_op_inputs.end(), while_cnode->inputs().begin() + kWhileMinInputSize,
                                  while_cnode->inputs().end());
    body_partial_op_inputs.insert(body_partial_op_inputs.end(), while_cnode->inputs().begin() + kWhileMinInputSize,
                                  while_cnode->inputs().end());

    static int idx = 0;
    auto cond_partial_node = graph->NewCNode(cond_partial_op_inputs);
    cond_partial_node->set_fullname_with_scope("Partial-while-cond-" + std::to_string(idx));
    cond_partial_node->set_abstract(cond_fg->output()->abstract());

    auto body_partial_node = graph->NewCNode(body_partial_op_inputs);
    body_partial_node->set_fullname_with_scope("Partial-while-body-" + std::to_string(idx));
    idx++;

    // concat body_fg output to cond_fg input
    auto body_output = body_fg->output();
    auto body_output_cnode = utils::cast<CNodePtr>(body_output);
    auto prim = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(body_output_cnode->input(0));
    if (prim == nullptr) {
      MS_LOG(ERROR) << "Get PrimitiveC of node:" << body_output_cnode->fullname_with_scope() << " failed.";
      return false;
    }

    // concat body to cond
    std::vector<AnfNodePtr> body_to_cond_inputs{cond_vnode};
    if ((schema::PrimitiveType)(prim->Type()) == schema::PrimitiveType_MakeTuple) {
      for (size_t i = 1; i < body_output_cnode->inputs().size(); ++i) {
        body_to_cond_inputs.emplace_back(body_output_cnode->input(i));
      }
    } else {
      body_to_cond_inputs.emplace_back(body_output_cnode->input(1));
    }

    // concat body to cond
    auto body_to_cond_cnode = body_fg->NewCNode(body_to_cond_inputs);
    body_to_cond_cnode->set_fullname_with_scope("Partial-while-body-to-cond");
    auto body_fg_manager = body_fg->manager();
    body_fg_manager->Replace(body_fg->output(), body_to_cond_cnode);
    body_fg->set_output(body_to_cond_cnode);
    body_partial_node->set_abstract(cond_fg->output()->abstract());

    // create switch cnode
    ValueNodePtr switch_anf_primitive = GetSwitchAnfPrim();
    if (switch_anf_primitive == nullptr) {
      MS_LOG(ERROR) << "GetSwitchAnfPrim failed.";
      return false;
    }

    // insert switch node
    std::vector<AnfNodePtr> switch_op_inputs = {switch_anf_primitive, cond_partial_node, body_partial_node};
    auto switch_cnode = graph->NewCNode(switch_op_inputs);
    switch_cnode->set_fullname_with_scope("Switch-" + std::to_string(count++));

    AbstractBasePtrList abstract_list;
    auto body_fg_output_cnode = utils::cast<CNodePtr>(body_fg->output());
    for (auto &cnode : body_fg_output_cnode->inputs()) {
      if (!utils::isa<CNodePtr>(cnode) && !utils::isa<ParameterPtr>(cnode)) {
        continue;
      }
      abstract_list.push_back(cnode->abstract());
    }

    switch_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));

    // create cond partial cnode
    auto manager = graph->manager();
    auto node_users = manager->node_users()[while_cnode];
    for (auto &node_user : node_users) {
      manager->SetEdge(node_user.first, node_user.second, switch_cnode);
    }
  }

  return true;
}
}  // namespace mindspore::opt
