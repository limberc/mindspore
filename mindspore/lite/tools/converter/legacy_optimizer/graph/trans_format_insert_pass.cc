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

#include <string>
#include <memory>
#include <vector>
#include <utility>
#include "tools/converter/legacy_optimizer/graph/trans_format_insert_pass.h"
#include "tools/common/node_util.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"

namespace mindspore {
namespace {
std::vector<int> nchw2nhwc_perm = {0, 2, 3, 1};
std::vector<int> nhwc2nchw_perm = {0, 3, 1, 2};
}  // namespace
namespace lite {
bool TransOpInsertPass::CanFusion(schema::MetaGraphT *graph, const std::unique_ptr<CNodeT> &node) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(node != nullptr);
  auto input_node_indexes = GetInputNodeIdx(*graph, *node);
  pre_type_ = kNONE;
  size_t has_trans_count = 0;
  auto can_fusion = true;
  for (auto input_node_index : input_node_indexes) {
    MS_ASSERT(graph->nodes.size() > input_node_index);
    auto &pre_node = graph->nodes.at(input_node_index);
    MS_ASSERT(pre_node != nullptr);
    MS_ASSERT(pre_node->primitive != nullptr);
    MS_ASSERT(pre_node->primitive->value != nullptr);
    if (pre_type_ == kNONE) {
      if (pre_node->primitive->value.type == schema::PrimitiveType_Transpose) {
        MS_ASSERT(pre_node->primitive->value.AsTranspose() != nullptr);
        if (pre_node->primitive->value.AsTranspose()->perm == nchw2nhwc_perm) {
          pre_type_ = kNCHW2NHWC;
        } else if (pre_node->primitive->value.AsTranspose()->perm == nhwc2nchw_perm) {
          pre_type_ = kNHWC2NCHW;
        } else {
          return false;
        }
        has_trans_count++;
      }
    } else {
      if (pre_node->primitive->value.type == schema::PrimitiveType_Transpose) {
        auto cur_type = kNONE;
        if (pre_node->primitive->value.AsTranspose()->perm == nchw2nhwc_perm) {
          cur_type = kNCHW2NHWC;
        } else if (pre_node->primitive->value.AsTranspose()->perm == nhwc2nchw_perm) {
          cur_type = kNHWC2NCHW;
        } else {
          return false;
        }
        if (pre_type_ != cur_type) {
          can_fusion = false;
          break;
        } else {
          has_trans_count++;
        }
      }
    }
  }
  if (!can_fusion) {
    return false;
  }
  auto output_node_indexes = GetOutputNodeIdx(*graph, *node);
  post_type_ = kNONE;
  for (auto output_node_index : output_node_indexes) {
    MS_ASSERT(graph->nodes.size() > output_node_index);
    auto &post_node = graph->nodes.at(output_node_index);
    MS_ASSERT(post_node != nullptr);
    MS_ASSERT(post_node->primitive != nullptr);
    MS_ASSERT(post_node->primitive->value != nullptr);
    if (post_type_ == kNONE) {
      if (post_node->primitive->value.type == schema::PrimitiveType_Transpose) {
        if (post_node->primitive->value.AsTranspose()->perm == nchw2nhwc_perm) {
          post_type_ = kNCHW2NHWC;
        } else if (post_node->primitive->value.AsTranspose()->perm == nhwc2nchw_perm) {
          post_type_ = kNHWC2NCHW;
        } else {
          return false;
        }
        has_trans_count++;
      }
    } else {
      if (post_node->primitive->value.type == schema::PrimitiveType_Transpose) {
        auto cur_type = kNONE;
        if (post_node->primitive->value.AsTranspose()->perm == nchw2nhwc_perm) {
          cur_type = kNCHW2NHWC;
        } else if (post_node->primitive->value.AsTranspose()->perm == nhwc2nchw_perm) {
          cur_type = kNHWC2NCHW;
        } else {
          return false;
        }
        if (post_type_ != cur_type) {
          can_fusion = false;
          break;
        } else {
          has_trans_count++;
        }
      }
    }
  }
  if (!can_fusion) {
    return false;
  }
  if (pre_type_ == kNONE && post_type_ == kNONE) {
    return false;
  }
  auto output_size = output_node_indexes.empty() ? 1 : output_node_indexes.size();
  auto total_node_count = input_node_indexes.size() + output_size;
  size_t half_count = total_node_count / 2;
  if (GetCNodeTType(*node) == schema::PrimitiveType_Activation) {
    MS_ASSERT(node != nullptr);
    MS_ASSERT(node->primitive != nullptr);
    MS_ASSERT(node->primitive->value != nullptr);
    MS_ASSERT(node->primitive->value.AsActivation() != nullptr);
    if (node->primitive->value.AsActivation() != nullptr &&
        node->primitive->value.AsActivation()->type == schema::ActivationType_LEAKY_RELU) {
      return has_trans_count >= half_count;
    }
  }
  if (GetCNodeTType(*node) == schema::PrimitiveType_Split) {
    return has_trans_count >= half_count;
  }
  can_fusion = has_trans_count > half_count;
  return can_fusion;
}

STATUS TransOpInsertPass::FindOutTransType() {
  pre_insert_trans_type_ = kNHWC2NCHW;
  post_insert_trans_type_ = kNHWC2NCHW;
  if (pre_type_ == kNONE && post_type_ != kNONE) {
    pre_insert_trans_type_ = post_type_ == kNHWC2NCHW ? kNHWC2NCHW : kNCHW2NHWC;
    post_insert_trans_type_ = post_type_ == kNHWC2NCHW ? kNCHW2NHWC : kNHWC2NCHW;
  } else if (pre_type_ != kNONE && post_type_ == kNONE) {
    pre_insert_trans_type_ = pre_type_ == kNHWC2NCHW ? kNCHW2NHWC : kNHWC2NCHW;
    post_insert_trans_type_ = pre_type_ == kNHWC2NCHW ? kNHWC2NCHW : kNCHW2NHWC;
  } else if (pre_type_ == kNONE && post_type_ == kNONE) {
    MS_ASSERT(false);
  } else {
    if (pre_type_ == post_type_) {
      MS_LOG(ERROR) << "Unknow error";
      return RET_ERROR;
    }
    pre_insert_trans_type_ = pre_type_ == kNHWC2NCHW ? kNCHW2NHWC : kNHWC2NCHW;
    post_insert_trans_type_ = post_type_ == kNHWC2NCHW ? kNCHW2NHWC : kNHWC2NCHW;
  }
  return RET_OK;
}

STATUS TransOpInsertPass::Run(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  bool changed = true;
  int run_counts = 0;
  std::vector<CNodeT *> has_insert_nodes;
  while (changed && run_counts < 10) {
    changed = false;
    for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
      auto &node = *iter;
      if (node == nullptr || node->primitive == nullptr) {
        MS_LOG(ERROR) << "node or primitive null";
        return RET_NULL_PTR;
      }
      auto type = node->primitive->value.type;
      if (IsContain(has_insert_nodes, node.get()) || !IsContain(GetInsertOpList(), type)) {
        continue;
      }
      auto node_name = node->name;
      if (!CanFusion(graph, node)) {
        continue;
      }
      auto ret = FindOutTransType();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "FindOutTransType error";
        return ret;
      }
      ret = ChangeOpAxis(graph, node);
      if (ret == RET_NOT_SUPPORT) {
        MS_LOG(INFO) << "not support to ChangeOpAxis";
        return RET_OK;
      } else if (ret != RET_OK) {
        MS_LOG(INFO) << "no need to ChangeOpAxis";
        return ret;
      }
      has_insert_nodes.push_back(node.get());
      STATUS status = RET_OK;
      auto input_tensor_size = (*iter)->inputIndex.size();
      for (size_t i = 0; i < input_tensor_size; i++) {
#ifdef SUPPORT_TRAIN
        auto &tensor = graph->allTensors.at((*iter)->inputIndex[i]);
        MS_ASSERT(tensor != nullptr);
        if (tensor->nodeType == schema::NodeType_ValueNode) {
          continue;
        }
#endif
        auto &input_tensor = graph->allTensors.at((*iter)->inputIndex[i]);
        if (input_tensor->nodeType == NodeType_ValueNode && input_tensor->dims.size() < 4) {
          continue;
        }
        iter = InsertFormatTransNode(graph, iter, kBefore, i, pre_insert_trans_type_, &status);
        if (status != RET_OK) {
          MS_LOG(ERROR) << "Insert" << pre_insert_trans_type_ << "before " << (*iter)->name << " failed";
          return status;
        }
        if ((*iter)->primitive->value.type == schema::PrimitiveType_StridedSlice ||
            (*iter)->primitive->value.type == schema::PrimitiveType_Slice) {
          break;
        }
      }
      auto output_tensor_size = (*iter)->outputIndex.size();
      for (size_t i = 0; i < output_tensor_size; i++) {
        iter = InsertFormatTransNode(graph, iter, kAfter, i, post_insert_trans_type_, &status);
        if (status != RET_OK) {
          MS_LOG(ERROR) << "Insert" << post_insert_trans_type_ << "Node before " << (*iter)->name << " failed";
          return status;
        }
      }
      changed = true;
    }
    run_counts++;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
