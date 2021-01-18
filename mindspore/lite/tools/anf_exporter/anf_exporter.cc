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

#include "tools/anf_exporter/anf_exporter.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "src/ops/quant_dtype_cast.h"
#include "abstract/abstract_value.h"
#include "mindspore/core/ir/primitive.h"
#include "src/tensor.h"
#include "src/param_value_lite.h"
#include "src/common/utils.h"
#include "src/ops/partial.h"
#include "tools/common/graph_util.h"

namespace mindspore::lite {
void AnfExporter::RemoveIfMakeTuple(const CNodePtr &cnode) {
  bool has_make_tuple = false;
  std::vector<AnfNodePtr> inputs;
  inputs.clear();

  inputs.emplace_back(cnode->input(0));
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    AnfNodePtr input_node = cnode->input(i);
    if (!input_node->isa<CNode>()) {
      inputs.emplace_back(cnode->input(i));
      continue;
    }
    auto make_tuple_node = utils::cast<CNodePtr>(input_node);
    if (IsPrimitiveCNode(make_tuple_node, schema::PrimitiveType_MakeTuple)) {
      has_make_tuple = true;
      for (size_t j = 1; j < make_tuple_node->inputs().size(); ++j) {
        inputs.emplace_back(make_tuple_node->input(j));
      }
    } else {
      inputs.emplace_back(cnode->input(i));
    }
  }
  if (has_make_tuple) {
    cnode->set_inputs(inputs);
  }
}

void AnfExporter::RemoveIfDepend(const CNodePtr &cnode) {
  bool hasDepend = false;
  std::vector<AnfNodePtr> inputs;
  inputs.clear();

  inputs.emplace_back(cnode->input(0));
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    AnfNodePtr inputNode = cnode->input(i);
    if (!inputNode->isa<CNode>()) {
      inputs.emplace_back(cnode->input(i));
      continue;
    }
    auto dependNode = utils::cast<CNodePtr>(inputNode);
    if (IsPrimitiveCNode(dependNode, schema::PrimitiveType_Depend) ||
        IsPrimitiveCNode(dependNode, schema::PrimitiveType_ControlDepend)) {
      hasDepend = true;
      bool maskOut = (dependNode->inputs().size() == 3);
      for (size_t j = 1; j < dependNode->inputs().size(); ++j) {
        AnfNodePtr dependInputNode = dependNode->input(j);
        if (dependInputNode->isa<CNode>()) {
          inputs.emplace_back(dependInputNode);
          if (maskOut) {
            break;
          }
        }
      }
    } else {
      inputs.emplace_back(cnode->input(i));
    }
  }
  if (hasDepend) {
    cnode->set_inputs(inputs);
  }
}

int AnfExporter::ConvertQuantParam(const std::unique_ptr<schema::MetaGraphT> &meta_graph,
                                   const std::shared_ptr<PrimitiveC> &primitive,
                                   const std::unique_ptr<schema::CNodeT> &dst_node) {
  MS_ASSERT(meta_graph != nullptr);
  MS_ASSERT(primitive != nullptr);
  MS_ASSERT(dst_node != nullptr);
  // add quant param
  dst_node->quantType = primitive->quant_type();
  MS_LOG(DEBUG) << "node: " << dst_node->name << " add QuantParam";
  // activation
  auto input_quant_params = primitive->input_quant_params();
  auto node_type = (schema::PrimitiveType)primitive->Type();
  if (!input_quant_params.empty()) {
    for (size_t i = 0; i < input_quant_params.size(); i++) {
      if (i >= dst_node->inputIndex.size()) {
        MS_LOG(INFO) << "node: " << dst_node->name << " input has " << input_quant_params.size()
                     << " quant_params; but only " << dst_node->inputIndex.size() << " input";
        break;
      }
      auto activate_index = dst_node->inputIndex[i];
      auto tensor_input = meta_graph->allTensors[activate_index].get();
      if (tensor_input->quantParams.empty()) {
        for (auto input_quant_param : input_quant_params[i]) {
          std::unique_ptr<schema::QuantParamT> input_quant_param_ptr =
            std::make_unique<schema::QuantParamT>(input_quant_param);
          MS_LOG(DEBUG) << "[input][" << i << "]node: " << dst_node->name << " scale: " << input_quant_param_ptr->scale
                        << " zp: " << input_quant_param_ptr->zeroPoint;
          input_quant_param_ptr->dstDtype = tensor_input->dataType;
          tensor_input->quantParams.emplace_back(std::move(input_quant_param_ptr));
        }
      }
    }
  } else {
    MS_LOG(DEBUG) << "node: " << dst_node->name << " input quant params is empty";
  }
  // output

  auto output_quant_params = primitive->output_quant_params();
  if (output_quant_params.empty()) {
    if (node_type != schema::PrimitiveType_QuantDTypeCast) {
      MS_LOG(DEBUG) << "node: " << dst_node->name << " output quant params is empty";
    }
  } else {
    if (dst_node->outputIndex.size() != output_quant_params.size()) {
      MS_LOG(INFO) << "node: " << dst_node->name << " output has " << output_quant_params.size()
                   << " quant_params; but only " << dst_node->outputIndex.size() << " output";
      return RET_ERROR;
    }
    int output_idx = 0;
    for (const auto &output_quant_param : output_quant_params) {
      auto output_tensor = meta_graph->allTensors[dst_node->outputIndex[output_idx]].get();
      output_idx++;
      for (const auto &channel_quant_param : output_quant_param) {
        if (output_tensor->quantParams.empty() && dst_node->quantType != schema::QuantType_WeightQuant) {
          std::unique_ptr<schema::QuantParamT> output_quant_param_ptr =
            std::make_unique<schema::QuantParamT>(channel_quant_param);
          MS_LOG(DEBUG) << "[output]node: " << dst_node->name << " scale: " << output_quant_param_ptr->scale
                        << " zp: " << output_quant_param_ptr->zeroPoint;
          output_quant_param_ptr->dstDtype = output_tensor->dataType;
          output_tensor->quantParams.emplace_back(std::move(output_quant_param_ptr));
        }
      }
    }
  }

  auto first_output_index = dst_node->outputIndex[0];
  auto first_tensor_output = meta_graph->allTensors[first_output_index].get();
  if (dst_node->quantType == schema::QuantType_PostTraining) {
    if (node_type != schema::PrimitiveType_QuantDTypeCast) {
      first_tensor_output->dataType = kNumberTypeInt8;
    } else {
      MS_ASSERT(utils::isa<std::shared_ptr<QuantDTypeCast>>(primitive));
      auto primc = utils::cast<std::shared_ptr<QuantDTypeCast>>(primitive);
      MS_ASSERT(primc != nullptr);
      if (primc->GetDstT() != kNumberTypeFloat32) {
        first_tensor_output->dataType = kNumberTypeInt8;
      }
    }
  }
  return RET_OK;
}

std::vector<schema::CNodeT *> AnfExporter::GetSubgraphNodes(const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                                            const size_t &subgraph_index) {
  std::vector<schema::CNodeT *> subgraph_nodes{};
  subgraph_nodes.resize(meta_graphT->subGraph.at(subgraph_index)->nodeIndices.size());
  std::transform(meta_graphT->subGraph.at(subgraph_index)->nodeIndices.begin(),
                 meta_graphT->subGraph.at(subgraph_index)->nodeIndices.end(), subgraph_nodes.begin(),
                 [&meta_graphT](const uint32_t idx) { return meta_graphT->nodes.at(idx).get(); });
  return subgraph_nodes;
}

int AnfExporter::SetGraphInputIndex(const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                    const size_t &subgraph_index) {
  auto &subgraph = meta_graphT->subGraph.at(subgraph_index);
  auto subgraph_nodes = GetSubgraphNodes(meta_graphT, subgraph_index);
  std::vector<schema::CNodeT *> subgraph_input_nodes{};
  for (auto &node : subgraph_nodes) {
    if (IsContain(graph_input_nodes_, node)) {
      subgraph_input_nodes.push_back(node);
    }
  }
  std::vector<schema::TensorT *> subgraph_inputs{};
  for (auto &node : subgraph_input_nodes) {
    for (auto input : node->inputIndex) {
      auto tensor = meta_graphT->allTensors[input].get();
      if (tensor->nodeType != schema::NodeType_CNode && tensor->data.empty()) {
        tensor->nodeType = schema::NodeType_ValueNode;
        tensor->format = schema::Format_NHWC;
        if (!IsContain(subgraph->inputIndices, input)) {
          if (subgraph_index == kMainGraphIndex) {
            meta_graphT->inputIndex.push_back(input);
          }
          subgraph->inputIndices.push_back(input);
          subgraph_inputs.push_back(tensor);
        }
      }
    }
  }

  return RET_OK;
}

int AnfExporter::SetGraphoutputIndex(const CNodePtr &cnode, const size_t subgraph_index,
                                     const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                     const std::unique_ptr<schema::SubGraphT> &sub_graphT,
                                     schema::CNodeT *return_node) {
  MS_ASSERT(nullptr != meta_graphT);
  MS_ASSERT(nullptr != return_node);
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    if (input_node == nullptr) {
      MS_LOG(ERROR) << "output node is nullptr";
      return RET_NULL_PTR;
    } else if (input_node->isa<CNode>()) {
      auto ret = ConvertInputCNode(input_node, return_node);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "obtain outputs failed";
        return ret;
      }
    } else if (input_node->isa<Parameter>()) {
      MS_LOG(INFO) << "the node " << input_node->fullname_with_scope().c_str() << "is parameter node";
      continue;
    } else {
      MS_LOG(ERROR) << "the node " << input_node->fullname_with_scope().c_str() << "is not output node";
      return RET_ERROR;
    }
  }
  for (unsigned int &i : return_node->inputIndex) {
    if (subgraph_index == kMainGraphIndex) {
      meta_graphT->outputIndex.push_back(i);
    }
    meta_graphT->subGraph.at(subgraph_index)->outputIndices.push_back(i);
  }
  return RET_OK;
}

int AnfExporter::ExportSubgraph(const FuncGraphPtr &func_graph, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                const size_t &subgraph_index, bool keep_graph, bool copy_primitive,
                                const std::shared_ptr<AnfNode> &partial_anode) {
  int ret = RET_OK;
  meta_graphT->subGraph.emplace_back(std::make_unique<schema::SubGraphT>());
  auto &sub_graphT = meta_graphT->subGraph.at(subgraph_index);
  auto subgraph_name = func_graph->get_attr("graph_name");
  MS_ASSERT(subgraph_name != nullptr);
  sub_graphT->name = GetValue<std::string>(subgraph_name);
  auto fmk = func_graph->get_attr("fmk");
  MS_ASSERT(fmk != nullptr);
  meta_graphT->fmkType = GetValue<int>(fmk);

  auto cnodes = func_graph->GetOrderedCnodes();
  for (const auto &cnode : cnodes) {
    auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
    if (primitive_c == nullptr) {
      auto fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
      if (fg != nullptr) {
        auto partial_cnode = CreatePartialCnode(fg, cnode);
        primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(partial_cnode->input(0));
        auto primT = primitive_c->primitiveT();
        MS_ASSERT(primT != nullptr);
        auto pos = fg_subgraph_map.find(fg);
        if (pos != fg_subgraph_map.end()) {
          MS_ASSERT(primT->value.AsPartial() != nullptr);
          primT->value.AsPartial()->subGraphIndex = fg_subgraph_map.at(fg);
        } else {
          size_t next_subgraph_index = fg_subgraph_map.size() + 1;
          fg_subgraph_map.insert(std::pair<FuncGraphPtr, int>{fg, next_subgraph_index});
          MS_ASSERT(primT->value.AsPartial() != nullptr);
          primT->value.AsPartial()->subGraphIndex = next_subgraph_index;
          ret = ExportSubgraph(fg, meta_graphT, next_subgraph_index, keep_graph, copy_primitive, cnode);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "ExportSubgraph failed";
            break;
          }
        }
      } else {
        MS_LOG(ERROR) << "primitive_c is nullptr";
        ret = RET_MEMORY_FAILED;
        break;
      }
    }

#ifdef SUPPORT_TRAIN
    RemoveIfMakeTuple(cnode);
    RemoveIfDepend(cnode);
#endif

    if ((primitive_c->Type() == schema::PrimitiveType_TupleGetItem) ||
#ifdef SUPPORT_TRAIN
        (primitive_c->Type() == schema::PrimitiveType_Depend) ||
        (primitive_c->Type() == schema::PrimitiveType_ControlDepend) ||
#endif
        (primitive_c->Type() == schema::PrimitiveType_MakeTuple)) {
      continue;
    }
#ifndef SUPPORT_TRAIN
    RemoveIfMakeTuple(cnode);
#endif
    auto primT = primitive_c->primitiveT();
    auto node = std::make_unique<schema::CNodeT>();
    if (node == nullptr) {
      MS_LOG(ERROR) << "object failed to be constructed";
      ret = RET_MEMORY_FAILED;
      break;
    }
    if (primT->value.type == schema::PrimitiveType_Return) {
      node->name = "return_node";
      ret = SetGraphoutputIndex(cnode, subgraph_index, meta_graphT, sub_graphT, node.get());
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "SetOpOutputN failed";
        break;
      }
      continue;
    }

    node->nodeType = schema::NodeType_CNode;
    node->name = cnode->fullname_with_scope();
    if (copy_primitive) {
      auto primitive = new (std::nothrow) schema::PrimitiveT();
      if (primitive != nullptr) {
        *primitive = *primT;
        node->primitive = std::unique_ptr<schema::PrimitiveT>(primitive);
      }
    } else {
      node->primitive = std::unique_ptr<schema::PrimitiveT>(primT);
    }
    ret = SetOpInputNode(cnode, meta_graphT, node.get());
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "SetOpInputNode failed";
      break;
    }
    SetOpOutputNode(cnode, meta_graphT, node.get());
    ret = ConvertQuantParam(meta_graphT, primitive_c, node);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ConvertQuantParam failed";
      break;
    }
    if (!keep_graph) {
      primitive_c->ClearPrimitiveT();
    }
    meta_graphT->nodes.push_back(std::move(node));
    meta_graphT->subGraph.at(subgraph_index)->nodeIndices.push_back(node_idx++);
  }
  if (ret != RET_OK) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return ret;
  }

  ret = SetGraphInputIndex(meta_graphT, subgraph_index);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SetGraphInputIndex failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return ret;
  }

  ret = SetSubgraphTensorIndices(meta_graphT.get());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SetSubgraphTensorIndices failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return ret;
  }

  return RET_OK;
}

schema::MetaGraphT *AnfExporter::Export(const FuncGraphPtr &func_graph, bool keep_graph, bool copy_primitive) {
  static int subgraph_index = 0;
  auto meta_graphT = std::make_unique<schema::MetaGraphT>();
  int ret = ExportSubgraph(func_graph, meta_graphT, subgraph_index, keep_graph, copy_primitive);
  if (ret != RET_OK) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return nullptr;
  }
  return meta_graphT.release();
}

int AnfExporter::ConvertInputCNode(const std::shared_ptr<AnfNode> &input_anode, schema::CNodeT *output_cnode) {
  std::string input_name = input_anode->fullname_with_scope();
  auto input_cnode = utils::cast<CNodePtr>(input_anode);
  if (!IsPrimitiveCNode(input_cnode, schema::PrimitiveType_TupleGetItem)) {
#ifndef SUPPORT_TRAIN
    if (node_id_map_.find(input_name) != node_id_map_.end()) {
      output_cnode->inputIndex.emplace_back(node_id_map_[input_name]);
    }
#else
    bool found = false;
    if (node_id_map_.find(input_name) != node_id_map_.end()) {
      output_cnode->inputIndex.emplace_back(node_id_map_[input_name]);
      found = true;
    }

    if (found == false) {
      auto input_index_key = input_name + "_o:" + std::to_string(0);
      if (node_id_map_.find(input_index_key) != node_id_map_.end()) {
        output_cnode->inputIndex.emplace_back(node_id_map_[input_index_key]);
      }
    }
#endif
  } else {
    auto inputs = input_cnode->inputs();

    if (inputs.size() != 3) {
      MS_LOG(ERROR) << "TupleGetItem should have 3 inputs, got " << inputs.size();
      return RET_ERROR;
    }
    auto get_item_input_cnode = inputs.at(1);
    auto index_vnode = inputs.at(2);
    if (!utils::isa<ValueNode>(index_vnode)) {
      MS_LOG(ERROR) << "TupleGetItem's input 2 is not valuenode";
      return RET_ERROR;
    }
    auto value_node = utils::cast<ValueNodePtr>(index_vnode);
    if (value_node == nullptr) {
      MS_LOG(ERROR) << "cast to ValueNode failed";
      return RET_ERROR;
    }
    auto input_index_key = get_item_input_cnode->fullname_with_scope() + "_o:" +
                           std::to_string(value_node->value()->type()->number_type() == kNumberTypeInt64
                                            ? GetValue<int64_t>(value_node->value())
                                            : GetValue<int>(value_node->value()));
    auto iter = node_id_map_.find(input_index_key);
    if (iter == node_id_map_.end()) {
#ifdef SUPPORT_TRAIN
      input_index_key = get_item_input_cnode->fullname_with_scope() + "_o:" + std::to_string(0);  // try name with 0
      iter = node_id_map_.find(input_index_key);
      if (iter == node_id_map_.end()) {
        MS_LOG(ERROR) << "Can not find get_item output tensor " << input_index_key;
        return RET_ERROR;
      }
#else
      MS_LOG(ERROR) << "Can not find get_item output tensor " << input_index_key;
      return RET_ERROR;
#endif
    }
    output_cnode->inputIndex.emplace_back(iter->second);
  }
  return RET_OK;
}

int AnfExporter::ConvertInputParameter(const std::shared_ptr<AnfNode> &input_anode,
                                       const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                       schema::CNodeT *output_cnode) {
  auto paramNode = input_anode->cast<ParameterPtr>();
  std::string input_name = paramNode->fullname_with_scope();
  if (node_id_map_.find(input_name) != node_id_map_.end()) {
    output_cnode->inputIndex.emplace_back(node_id_map_[paramNode->name()]);
    return RET_OK;
  }
  auto paramTensor = std::make_unique<schema::TensorT>();
  paramTensor->format = schema::Format_NHWC;
  paramTensor->name = paramNode->name();
  auto abstractBase = paramNode->abstract();
  if (abstractBase == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << paramNode->name();
    return RET_PARAM_INVALID;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstractBase)) {
    MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << paramNode->name();
    return RET_INPUT_TENSOR_ERROR;
  }
  auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(abstractBase);
  auto typePtr = abstractTensor->element()->GetTypeTrack();
  MS_ASSERT(typePtr != nullptr);
  paramTensor->dataType = typePtr->type_id();
  if (!utils::isa<abstract::ShapePtr>(abstractTensor->BuildShape())) {
    MS_LOG(ERROR) << "Shape of Abstract of parameter should be ShapePtr, " << paramNode->name();
    return RET_PARAM_INVALID;
  }
  auto shape_vector = utils::cast<abstract::ShapePtr>(abstractTensor->BuildShape())->shape();
  std::vector<int32_t> dims;
  (void)std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(dims),
                       [](const int64_t &value) { return static_cast<int32_t>(value); });
  paramTensor->dims = dims;
  auto paramValue = std::dynamic_pointer_cast<ParamValueLite>(paramNode->default_param());
  if (paramValue != nullptr && paramValue->tensor_size() != 0) {
    paramTensor->data.resize(paramValue->tensor_size());
    paramTensor->format = schema::Format(paramValue->format());
    if (EOK != memcpy_s(paramTensor->data.data(), paramTensor->data.size(), paramValue->tensor_addr(),
                        paramValue->tensor_size())) {
      MS_LOG(ERROR) << "memcpy_s failed.";
      return RET_ERROR;
    }
  }

  paramTensor->name = input_name;
  node_id_map_[input_name] = meta_graphT->allTensors.size();
  output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
  meta_graphT->allTensors.emplace_back(std::move(paramTensor));
  return RET_OK;
}

int AnfExporter::ConvertInputValueNode(const std::shared_ptr<AnfNode> &input_anode,
                                       const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                       schema::CNodeT *output_cnode) {
  auto valueNode = input_anode->cast<ValueNodePtr>();
  auto paramTensor = std::make_unique<schema::TensorT>();
  auto value = valueNode->value();
  if (value->isa<tensor::Tensor>()) {
    auto valueAbstract = valueNode->abstract();
    auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(valueAbstract);
    if (abstractTensor == nullptr || abstractTensor->element() == nullptr) {
      MS_LOG(ERROR) << "abstractTensor or abstractTensor->element() is nullptr";
      return RET_ERROR;
    }
    auto typePtr = abstractTensor->element()->GetTypeTrack();
    paramTensor->dataType = typePtr->type_id();
    auto shape_vector = utils::cast<abstract::ShapePtr>(abstractTensor->BuildShape())->shape();
    std::vector<int32_t> dims;
    (void)std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(dims),
                         [](const int64_t &value) { return static_cast<int32_t>(value); });
    paramTensor->dims = dims;
#ifdef SUPPORT_TRAIN
    if (paramTensor->dims.size() == 0) paramTensor->dims = {1};
    paramTensor->name = valueNode->fullname_with_scope();
#endif
    paramTensor->nodeType = schema::NodeType::NodeType_ValueNode;
    auto data = value->cast<tensor::TensorPtr>();
    paramTensor->data.resize(data->Size());
    auto ret = memcpy_s(paramTensor->data.data(), data->Size(), data->data_c(), data->Size());
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s error.";
      return RET_ERROR;
    }
    node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
    output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
    meta_graphT->allTensors.emplace_back(std::move(paramTensor));
  } else if (value->isa<mindspore::Int32Imm>() || value->isa<mindspore::Int64Imm>()) {
    auto valueAbstract = valueNode->abstract();
    auto abstractScalar = utils::cast<abstract::AbstractScalarPtr>(valueAbstract);
    auto typePtr = abstractScalar->GetTypeTrack();
    // data of int64 is converted to int32 here.
    paramTensor->dataType = kNumberTypeInt32;
    paramTensor->dims = {1};
    paramTensor->nodeType = schema::NodeType::NodeType_ValueNode;
    int real_data = CastToInt(value).front();
    paramTensor->data.resize(sizeof(int32_t));
    auto ret = memcpy_s(paramTensor->data.data(), sizeof(int32_t), &real_data, sizeof(int32_t));
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s error.";
      return RET_ERROR;
    }
    node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
    output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
    meta_graphT->allTensors.emplace_back(std::move(paramTensor));
  } else if (value->isa<mindspore::BoolImm>()) {
    auto valueAbstract = valueNode->abstract();
    auto abstractScalar = utils::cast<abstract::AbstractScalarPtr>(valueAbstract);
    auto typePtr = abstractScalar->GetTypeTrack();
    paramTensor->dataType = typePtr->type_id();
    paramTensor->dims = {1};
    paramTensor->nodeType = schema::NodeType_ValueNode;
    auto data = value->cast<mindspore::BoolImmPtr>();
    paramTensor->data.emplace_back(data->value());
    node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
    output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
    meta_graphT->allTensors.emplace_back(std::move(paramTensor));
  } else if (value->isa<mindspore::Int>()) {
    paramTensor->dataType = kNumberTypeInt32;
    paramTensor->dims = {1};
    paramTensor->nodeType = schema::NodeType_ValueNode;
    paramTensor->data.emplace_back(kNumberTypeInt32);
    node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
    output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
    meta_graphT->allTensors.emplace_back(std::move(paramTensor));
  } else if (value->isa<mindspore::ValueSequeue>()) {
    auto valueAbstract = valueNode->abstract();
    auto abstractSequnce = utils::cast<abstract::AbstractSequeuePtr>(valueAbstract);
    if (abstractSequnce->isa<abstract::AbstractTuple>()) {
      auto abstractTuple = utils::cast<abstract::AbstractTuplePtr>(valueAbstract);
      auto x_shape_data = abstractTuple->elements();
      std::vector<int32_t> shape;
      for (std::size_t i = 0; i < abstractTuple->size(); ++i) {
        auto value_track = x_shape_data[i]->GetValueTrack();
        MS_ASSERT(value_track != nullptr);
        if (value_track->isa<Int32Imm>()) {
          shape.push_back((GetValue<int>(value_track)));
        } else if (value_track->isa<Int64Imm>()) {
          shape.push_back((GetValue<int64_t>(value_track)));
        } else {
          MS_LOG(ERROR) << "Value type is ValueSequence is not integer, it is " << value_track->ToString() << ".";
          return RET_ERROR;
        }
      }
      auto typePtr = abstractTuple->elements()[0]->GetTypeTrack();
      paramTensor->dataType = kNumberTypeInt32;
      paramTensor->dims = {static_cast<int32_t>(shape.size())};
      paramTensor->nodeType = schema::NodeType_ValueNode;
      paramTensor->data.resize(shape.size() * sizeof(int));
      auto ret = memcpy_s(paramTensor->data.data(), shape.size() * sizeof(int32_t), shape.data(),
                          shape.size() * sizeof(int32_t));
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "memcpy_s data into paramTensor failed.";
        return RET_ERROR;
      }
      node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
      output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
      meta_graphT->allTensors.emplace_back(std::move(paramTensor));
    }
  } else if (value->isa<mindspore::BoolImm>()) {
    auto valueAbstract = valueNode->abstract();
    auto abstractScalar = utils::cast<abstract::AbstractScalarPtr>(valueAbstract);
    auto typePtr = abstractScalar->GetTypeTrack();
    paramTensor->dataType = typePtr->type_id();
    paramTensor->dims = {1};
    paramTensor->nodeType = schema::NodeType_ValueNode;
    auto data = value->cast<mindspore::BoolImmPtr>();
    paramTensor->data.emplace_back(data->value());
    node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
    output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
    meta_graphT->allTensors.emplace_back(std::move(paramTensor));
  } else if (value->isa<Number>()) {
    MS_LOG(INFO) << "Value is a number.";
    return RET_OK;
  } else if (value->isa<mindspore::ParamValueLite>()) {
    auto valueLite = std::dynamic_pointer_cast<ParamValueLite>(value);
    paramTensor->data.resize(valueLite->tensor_size());
    paramTensor->format = schema::Format(valueLite->format());
    paramTensor->dataType = valueLite->tensor_type();
    paramTensor->dims = valueLite->tensor_shape();
#ifdef SUPPORT_TRAIN
    if (paramTensor->dims.size() == 0) {
      paramTensor->dims = {1};
    }
#endif
    auto ret = memcpy_s(paramTensor->data.data(), valueLite->tensor_size() * sizeof(uint8_t), valueLite->tensor_addr(),
                        valueLite->tensor_size());
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s data into tensor failed.";
      return RET_ERROR;
    }
    node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
    output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
    meta_graphT->allTensors.emplace_back(std::move(paramTensor));
  } else if (value->isa<FuncGraph>()) {
    MS_LOG(INFO) << "op name:" << input_anode->fullname_with_scope() << " input is func_graph";
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "Not support value type , need add support.";
    return RET_ERROR;
  }
  return RET_OK;
}

int AnfExporter::SetOpInputNode(const CNodePtr &cnode, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                schema::CNodeT *fb_node) {
  MS_ASSERT(nullptr != meta_graphT);
  MS_ASSERT(nullptr != fb_node);
  if (cnode->inputs().size() <= 1) {
    return RET_OK;
  }
  bool is_graph_input = false;
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    if (input_node->isa<CNode>()) {
      auto ret = ConvertInputCNode(input_node, fb_node);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "ConvertInputCNode failed";
        return ret;
      }
    } else if (input_node->isa<Parameter>()) {
      auto ret = ConvertInputParameter(input_node, meta_graphT, fb_node);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "ConvertInputParameter failed";
        return ret;
      }
      if (!input_node->cast<ParameterPtr>()->has_default()) {
        is_graph_input = true;
      }
    } else if (input_node->isa<ValueNode>()) {
      auto ret = ConvertInputValueNode(input_node, meta_graphT, fb_node);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "ConvertInputValueNode failed";
        return RET_ERROR;
      }
    }
  }
  fb_node->name = cnode->fullname_with_scope();
  if (is_graph_input) {
    graph_input_nodes_.emplace_back(fb_node);
  }
  return RET_OK;
}

void AnfExporter::SetOpOutputNode(const CNodePtr &cnode, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                  schema::CNodeT *fb_node) {
  MS_ASSERT(nullptr != meta_graphT);
  MS_ASSERT(nullptr != fb_node);
  std::string cnode_name = fb_node->name;

  if (utils::isa<abstract::AbstractTuple>(cnode->abstract())) {
    auto tuple = std::reinterpret_pointer_cast<abstract::AbstractTuple>(cnode->abstract());
    if (tuple == nullptr) {
      MS_LOG(ERROR) << "tuple is nullptr";
      return;
    }
    auto elements = tuple->elements();
    for (size_t i = 0; i < elements.size(); i++) {
      auto msTensor = new (std::nothrow) schema::TensorT();
      if (msTensor == nullptr) {
        MS_LOG(ERROR) << "new msTensor failed";
        return;
      }
      msTensor->nodeType = schema::NodeType_CNode;
      fb_node->outputIndex.emplace_back(meta_graphT->allTensors.size());
#ifdef SUPPORT_TRAIN
      std::string name = cnode_name + "_o:" + std::to_string(i);
      node_id_map_[name] = meta_graphT->allTensors.size();
      meta_graphT->allTensors.emplace_back(msTensor);
      if (IsPrimitiveCNode(cnode, schema::PrimitiveType_Conv2D) ||
          IsPrimitiveCNode(cnode, schema::PrimitiveType_DepthwiseConv2D) ||
          IsPrimitiveCNode(cnode, schema::PrimitiveType_Adam))
        break;
#else
      if (elements.size() == 1) {
        node_id_map_[cnode_name] = meta_graphT->allTensors.size();
        msTensor->name = cnode_name;
      } else {
        std::string name = cnode_name + "_o:" + std::to_string(i);
        node_id_map_[name] = meta_graphT->allTensors.size();
        msTensor->name = name;
      }

      if (!utils::isa<abstract::AbstractTensorPtr>(elements[i])) {
        MS_LOG(ERROR) << "abstract is not AbstractTensor";
        delete (msTensor);
        return;
      }
      auto type = kNumberTypeFloat32;
      if (utils::isa<abstract::AbstractTensorPtr>(elements[i])) {
        auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(elements[i]);
        auto typePtr = abstract_tensor->element()->GetTypeTrack();
        type = typePtr->type_id();
      }
      msTensor->dataType = type;
      meta_graphT->allTensors.emplace_back(msTensor);
      if (IsPrimitiveCNode(cnode, schema::PrimitiveType_Conv2D) ||
          IsPrimitiveCNode(cnode, schema::PrimitiveType_DepthwiseConv2D) ||
          IsPrimitiveCNode(cnode, schema::PrimitiveType_FusedBatchNorm) ||
          IsPrimitiveCNode(cnode, schema::PrimitiveType_LayerNorm)) {
        break;
      }
#endif
    }
  } else {
    auto ms_tensor = new (std::nothrow) schema::TensorT();
    if (ms_tensor == nullptr) {
      MS_LOG(ERROR) << "new tensor failed";
      return;
    }
    auto type = kNumberTypeFloat32;
    if (utils::isa<abstract::AbstractTensorPtr>(cnode->abstract())) {
      auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(cnode->abstract());
      auto typePtr = abstract_tensor->element()->GetTypeTrack();
      type = typePtr->type_id();
    }
    ms_tensor->dataType = type;
    ms_tensor->nodeType = schema::NodeType_CNode;
    ms_tensor->name = cnode_name;
    fb_node->outputIndex.emplace_back(meta_graphT->allTensors.size());
    node_id_map_[cnode_name] = meta_graphT->allTensors.size();
    meta_graphT->allTensors.emplace_back(ms_tensor);
  }
}

bool AnfExporter::HasPrimitiveCNode(const AnfNodePtr &node) {
  MS_ASSERT(node != nullptr);
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return false;
  }

  auto prim = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
  if (prim == nullptr) {
    return false;
  }
  return true;
}

bool AnfExporter::IsPrimitiveCNode(const AnfNodePtr &node, schema::PrimitiveType type) {
  MS_ASSERT(node != nullptr);
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return false;
  }

  auto prim = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
  if (prim == nullptr) {
    return false;
  }
  return (schema::PrimitiveType)(prim->Type()) == type;
}

ValueNodePtr AnfExporter::GetPartialAnfPrim() {
  auto partial_primitiveT = new (std::nothrow) schema::PrimitiveT;
  if (partial_primitiveT == nullptr) {
    MS_LOG(ERROR) << "new partial_primitiveT failed";
    return nullptr;
  }
  partial_primitiveT->value.type = schema::PrimitiveType_Partial;
  partial_primitiveT->value.value = new (std::nothrow) schema::PartialT;
  if (partial_primitiveT->value.value == nullptr) {
    MS_LOG(ERROR) << "new PartialT failed";
    delete (partial_primitiveT);
    return nullptr;
  }

  auto partial_prim = std::make_shared<lite::Partial>(partial_primitiveT);
  ValueNodePtr partial_anf_prim = NewValueNode(partial_prim);
  return partial_anf_prim;
}

CNodePtr AnfExporter::CreatePartialCnode(const FuncGraphPtr &fg, AnfNodePtr node) {
  if (utils::isa<CNodePtr>(node)) {
    auto cnode = utils::cast<CNodePtr>(node);
    auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
    if (primitive_c != nullptr) {
      return cnode;
    }
    auto partial_anf_prim_vnode = GetPartialAnfPrim();
    auto cnode_input = cnode->inputs();
    cnode_input.insert(cnode_input.begin(), partial_anf_prim_vnode);
    cnode->set_inputs(cnode_input);
    return cnode;
  } else if (utils::isa<ValueNodePtr>(node)) {
    auto partial_anf_prim_vnode = GetPartialAnfPrim();
    std::vector<AnfNodePtr> inputs{partial_anf_prim_vnode, node};
    auto cnode = fg->NewCNode(inputs);
    return cnode;
  } else {
    MS_LOG(ERROR) << "failed to create partial cnode.";
    return nullptr;
  }
}

schema::MetaGraphT *Export(const FuncGraphPtr &func_graph, bool keep_graph, bool copy_primitive) {
  AnfExporter anf_exporter;
  return anf_exporter.Export(func_graph, keep_graph, copy_primitive);
}
}  // namespace mindspore::lite
