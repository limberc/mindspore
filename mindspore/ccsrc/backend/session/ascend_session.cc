/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include "backend/session/ascend_session.h"
#include <algorithm>
#include <map>
#include <tuple>
#include <set>
#include <string>
#include <list>

#include "base/core_ops.h"
#include "base/base_ref_utils.h"
#include "ir/tensor.h"
#include "ir/anf.h"
#include "common/trans.h"
#include "runtime/device/kernel_runtime.h"
#include "runtime/device/ascend/kernel_select_ascend.h"
#include "runtime/device/ascend/kernel_build_ascend.h"
#include "runtime/device/ascend/ascend_kernel_runtime.h"
#include "backend/optimizer/ascend/ascend_backend_optimization.h"
#include "backend/optimizer/common/common_backend_optimization.h"
#include "backend/optimizer/ascend/mindir/dropout_unify_mindir.h"
#include "backend/optimizer/ascend/mindir/maxpool_to_maxpool_with_argmax.h"
#include "backend/optimizer/ascend/mindir/maxpool_with_argmax_unify_mindir.h"
#include "backend/optimizer/ascend/mindir/conv2d_unify_mindir.h"
#include "backend/optimizer/ascend/mindir/sparse_softmax_cross_entropy_with_logits_unify_mindir.h"
#include "runtime/device/kernel_adjust.h"
#include "runtime/device/ascend/ascend_stream_assign.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/ms_utils.h"
#include "backend/optimizer/common/helper.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "utils/config_manager.h"
#include "debug/data_dump/dump_json_parser.h"
#include "debug/tensor_load.h"
#include "debug/anf_ir_utils.h"
#include "backend/optimizer/graph_kernel/reorder_ops.h"
#include "backend/optimizer/graph_kernel/basic_ops_fusion.h"
#include "backend/optimizer/graph_kernel/eliminate_redundant_output.h"
#include "backend/optimizer/graph_kernel/tensor_promotion.h"
#include "backend/optimizer/graph_kernel/graph_kernel_splitter.h"
#include "backend/optimizer/graph_kernel/graph_kernel_expander.h"
#include "backend/optimizer/graph_kernel/graph_kernel_cse.h"
#include "backend/optimizer/graph_kernel/value_graph_binder.h"
#include "backend/optimizer/graph_kernel/add_atomic_clean.h"
#include "backend/optimizer/pass/getitem_tuple.h"
#include "debug/data_dump/e2e_dump_util.h"
#include "debug/anf_ir_dump.h"
#include "debug/dump_proto.h"
#include "toolchain/adx_datadump_server.h"
#if ENABLE_CPU && ENABLE_D
#include "ps/util.h"
#include "ps/ps_cache/ps_cache_manager.h"
#endif
static constexpr uint32_t kLabelSwitchLabelId = 2;
namespace mindspore {
namespace session {
const size_t kInvalidIndex = SIZE_MAX;
constexpr size_t kReturnDataIndex = 1;
constexpr char SR_TAG[] = "sr_tag";
constexpr char BACKWARD[] = "backward";
namespace {
void DumpGraphExeOrder(const std::vector<CNodePtr> &execution_order, const std::string &tag = "") {
  MS_LOG(INFO) << "Dump execution_order size " << execution_order.size();
  MS_LOG(INFO) << "[index][stream_label][graph_id][node string]";
  int i = 0;
  for (auto &cnode : execution_order) {
    MS_EXCEPTION_IF_NULL(cnode);
    MS_LOG(INFO) << "[ " << i << "]"
                 << "[" << AnfAlgo::GetStreamDistinctionLabel(cnode.get()) << "]"
                 << "[" << AnfAlgo::GetGraphId(cnode.get()) << "]"
                 << "[" << cnode->DebugString() << "]";
    i++;
  }

  std::stringstream buf;
  buf << "================== execution order ==================\n";
  if (!tag.empty()) {
    buf << tag << "\n";
  }
  buf << "execution_order size: " << execution_order.size() << "\n";
  i = 0;
  for (auto &cnode : execution_order) {
    MS_EXCEPTION_IF_NULL(cnode);
    buf << i << ":\n";
    buf << "\t" << cnode->DebugString() << "\n";
    buf << "\t" << AnfAlgo::GetStreamDistinctionLabel(cnode.get()) << "\n";
    buf << "\t" << AnfAlgo::GetGraphId(cnode.get()) << "\n";
    i++;
  }
  buf << "================== execution order ==================\n";
}

void SetStreamDistinctionLabel(const KernelGraphPtr &graph, uint32_t label, bool is_override) {
  MS_EXCEPTION_IF_NULL(graph);
  if (is_override || graph->stream_distinction_label() == kInvalidDistincLabel) {
    graph->set_stream_distinction_label(label);
  }
}

std::vector<CNodePtr> GetCNodes(const std::vector<AnfNodePtr> &anf_nodes) {
  std::vector<CNodePtr> cnodes = {};
  size_t i = 0;
  for (const auto &anf : anf_nodes) {
    MS_LOG(INFO) << "Apply_list[" << i++ << "] = " << anf->DebugString();
    MS_EXCEPTION_IF_NULL(anf);
    if (anf->isa<CNode>()) {
      cnodes.push_back(anf->cast<CNodePtr>());
    }
  }
  return cnodes;
}

void InsertMakeTupleForOutput(NotNull<KernelGraphPtr> root_graph) {
  auto return_node = root_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  if (return_node->size() <= kReturnDataIndex) {
    return;
  }
  auto make_tuple = root_graph->NewCNode(
    {NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name())), root_graph->output()});
  root_graph->set_output(make_tuple);
}

BaseRef CreateNodeOutputPlaceholder(const session::KernelWithIndex &node_output_pair, const KernelGraphPtr &graph,
                                    const std::vector<tensor::TensorPtr> &input_tensors,
                                    const std::vector<size_t> &indexes,
                                    std::map<KernelWithIndex, std::vector<std::vector<size_t>>> *output_indexes) {
  auto &node = node_output_pair.first;
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(output_indexes);
  MS_LOG(INFO) << "Create placeholder for output[" << node->DebugString() << "] index[" << node_output_pair.second
               << "]";
  // if node is a value node, no need sync addr from device to host
  if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    return value_node->value();
  }
  if (node->isa<Parameter>()) {
    for (size_t input_idx = 0; input_idx < graph->inputs().size(); input_idx++) {
      if (input_idx >= input_tensors.size()) {
        MS_LOG(EXCEPTION) << "Input idx:" << input_idx << "out of range:" << input_tensors.size();
      }
      if (graph->inputs()[input_idx] == node) {
        return input_tensors[input_idx];
      }
    }
    MS_LOG(EXCEPTION) << "Parameter: " << node->DebugString() << " has no output addr";
  }
  (*output_indexes)[node_output_pair].emplace_back(indexes);
  BaseRef output_placeholder = std::make_shared<BaseRef>();
  return output_placeholder;
}

BaseRef CreateNodeOutputPlaceholder(const AnfNodePtr &anf, const KernelGraphPtr &graph,
                                    const std::vector<tensor::TensorPtr> &input_tensors,
                                    const std::vector<size_t> &indexes,
                                    std::map<KernelWithIndex, std::vector<std::vector<size_t>>> *output_indexes) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(output_indexes);
  MS_LOG(INFO) << "Create placeholder for output[" << anf->DebugString() << "]";
  auto item_with_index = AnfAlgo::VisitKernelWithReturnType(anf, 0);
  MS_EXCEPTION_IF_NULL(item_with_index.first);
  MS_LOG(INFO) << "Create placeholder for output after visit:" << item_with_index.first->DebugString();
  // special handle for maketuple
  if (AnfAlgo::CheckPrimitiveType(item_with_index.first, prim::kPrimMakeTuple)) {
    auto cnode = item_with_index.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    VectorRef ret;
    for (size_t i = 1; i < cnode->inputs().size(); ++i) {
      std::vector<size_t> cur_index = indexes;
      cur_index.emplace_back(i - 1);
      auto out = CreateNodeOutputPlaceholder(cnode->input(i), graph, input_tensors, cur_index, output_indexes);
      ret.push_back(out);
    }
    return ret;
  }
  // if is graph return nothing ,the function should return a null anylist
  size_t size = AnfAlgo::GetOutputTensorNum(item_with_index.first);
  if (size == 0) {
    return VectorRef();
  }
  return CreateNodeOutputPlaceholder(item_with_index, graph, input_tensors, indexes, output_indexes);
}

void CreateOutputPlaceholder(const KernelGraphPtr &kernel_graph, const std::vector<tensor::TensorPtr> &input_tensors,
                             VectorRef *outputs,
                             std::map<KernelWithIndex, std::vector<std::vector<size_t>>> *output_indexes) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(output_indexes);
  auto anf_outputs = kernel_graph->outputs();
  size_t index = 0;
  for (auto &item : anf_outputs) {
    MS_EXCEPTION_IF_NULL(item);
    MS_LOG(INFO) << "Create node output placeholder[" << item->DebugString() << "]";
    std::vector<size_t> indexes{index++};
    outputs->emplace_back(CreateNodeOutputPlaceholder(item, kernel_graph, input_tensors, indexes, output_indexes));
  }
}

void GetRefCount(KernelGraph *graph, std::map<KernelWithIndex, size_t> *ref_count) {
  MS_EXCEPTION_IF_NULL(graph);
  for (const auto &kernel : graph->execution_order()) {
    for (size_t i = 1; i < kernel->inputs().size(); i += 1) {
      const auto &input = kernel->input(i);
      auto kernel_with_index = AnfAlgo::VisitKernel(input, 0);
      const auto &node = kernel_with_index.first;
      if (node->isa<CNode>()) {
        (*ref_count)[kernel_with_index] += 1;
      }
    }
  }
}

void GetParameterIndex(KernelGraph *graph, const std::vector<tensor::TensorPtr> &inputs,
                       std::map<AnfNodePtr, size_t> *parameter_index) {
  size_t index = 0;
  for (const auto &input_node : graph->inputs()) {
    auto params = AnfAlgo::GetAllOutput(input_node);
    for (const auto &param : params) {
      if (index >= inputs.size()) {
        MS_LOG(EXCEPTION) << "Parameter size out of range. Parameter index: " << index
                          << ", input size: " << inputs.size();
      }
      const auto &input = inputs[index];
      // Check shape of input and parameter
      const auto &input_shape = input->shape();
      const auto &param_shape = AnfAlgo::GetOutputInferShape(param, 0);
      if (input_shape.size() != param_shape.size()) {
        MS_LOG(EXCEPTION) << "Shapes of input and parameter are different, input index: " << index
                          << ", parameter: " << param->fullname_with_scope();
      }
      for (size_t i = 0; i < input_shape.size(); i += 1) {
        if (input_shape[i] < 0 || static_cast<size_t>(input_shape[i]) != param_shape[i]) {
          MS_LOG(EXCEPTION) << "Shapes of input and parameter are different, input index: " << index
                            << ", parameter: " << param->fullname_with_scope();
        }
      }
      parameter_index->emplace(param, index++);
    }
  }
}

void GetOpInputTensors(const CNodePtr &cnode, const std::map<KernelWithIndex, tensor::TensorPtr> &op_output,
                       const std::map<AnfNodePtr, size_t> &parameter_index,
                       const std::vector<tensor::TensorPtr> &graph_inputs, InputTensorInfo *input_tensor_info) {
  MS_EXCEPTION_IF_NULL(cnode);
  for (size_t i = 1; i < cnode->inputs().size(); i += 1) {
    const auto &input = cnode->input(i);
    auto kernel_with_index = AnfAlgo::VisitKernel(input, 0);
    auto real_input = kernel_with_index.first;
    MS_EXCEPTION_IF_NULL(real_input);
    tensor::TensorPtr tensor = nullptr;
    if (real_input->isa<ValueNode>()) {
      auto value_node = real_input->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      auto value = GetValueNode(value_node);
      MS_EXCEPTION_IF_NULL(value);
      if (value->isa<ValueTuple>()) {
        auto value_tuple = value->cast<ValueTuplePtr>();
        MS_EXCEPTION_IF_NULL(value_tuple);
        if (kernel_with_index.second >= value_tuple->size()) {
          MS_LOG(EXCEPTION) << "Index " << kernel_with_index.second << "is out of value tuple range";
        }
        auto tensor_value = value_tuple->value()[kernel_with_index.second];
        if (tensor_value->isa<tensor::Tensor>()) {
          tensor = tensor_value->cast<tensor::TensorPtr>();
        }
      } else if (value->isa<tensor::Tensor>()) {
        if (kernel_with_index.second != 0) {
          MS_LOG(EXCEPTION) << "Index should be 0 for Tensor ValueNode, but is " << kernel_with_index.second;
        }
        tensor = GetValueNode<TensorPtr>(value_node);
      }
    } else if (real_input->isa<Parameter>()) {
      const auto &iter = parameter_index.find(real_input);
      if (iter == parameter_index.end()) {
        MS_LOG(EXCEPTION) << "Can not find parameter input of cnode, node = " << cnode->DebugString();
      }
      const size_t index = iter->second;
      if (index >= graph_inputs.size()) {
        MS_LOG(EXCEPTION) << "Parameter index is greater than size of graph's input tensor, parameter index = "
                          << cnode->DebugString() << "input tensor size = " << graph_inputs.size();
      }
      tensor = graph_inputs[index];
    } else if (real_input->isa<CNode>()) {
      const auto &iter = op_output.find(kernel_with_index);
      if (iter == op_output.end()) {
        MS_LOG(EXCEPTION) << "Can not find output tensor of cnode, node = " << real_input->DebugString();
      }
      tensor = iter->second;
      input_tensor_info->input_kernel.insert(kernel_with_index);
    } else {
      MS_LOG(EXCEPTION) << "Invalid input node, node = " << real_input->DebugString();
    }
    MS_EXCEPTION_IF_NULL(tensor);
    MS_LOG(DEBUG) << "Get" << i << "th input tensor of " << cnode->fullname_with_scope() << " from "
                  << real_input->fullname_with_scope() << "-" << kernel_with_index.second;
    input_tensor_info->input_tensors_mask.emplace_back(tensor->is_parameter() ? kParameterWeightTensorMask
                                                                              : kParameterDataTensorMask);
    input_tensor_info->input_tensors.emplace_back(tensor);
  }
}

void HandleOpInputs(const std::set<KernelWithIndex> &input_kernel, std::map<KernelWithIndex, size_t> *ref_count,
                    std::map<KernelWithIndex, tensor::TensorPtr> *op_output_map) {
  for (auto &kernel_with_index : input_kernel) {
    if (!kernel_with_index.first->isa<CNode>()) {
      continue;
    }
    auto ref_iter = ref_count->find(kernel_with_index);
    if (ref_iter == ref_count->end()) {
      MS_LOG(EXCEPTION) << "Can not find input KernelWithIndex in cnode reference count map, input cnode = "
                        << kernel_with_index.first->DebugString() << ", index = " << kernel_with_index.second;
    }
    // Reduce reference count number, when it was reduced to zero, release the useless output of pre node.
    ref_iter->second -= 1;
    if (ref_iter->second != 0) {
      continue;
    }
    ref_count->erase(ref_iter);
    auto output_iter = op_output_map->find(kernel_with_index);
    if (output_iter == op_output_map->end()) {
      MS_LOG(EXCEPTION) << "Can not find input KernelWithIndex in op_output map, input cnode = "
                        << kernel_with_index.first->DebugString() << ", index = " << kernel_with_index.second;
    }
    op_output_map->erase(output_iter);
  }
}

void HandleOpOutputs(const AnfNodePtr &kernel, const VectorRef &op_outputs,
                     const std::map<KernelWithIndex, std::vector<std::vector<size_t>>> &output_indexes,
                     const std::map<KernelWithIndex, size_t> &ref_count,
                     std::map<KernelWithIndex, tensor::TensorPtr> *op_output_map, VectorRef *outputs) {
  auto output_tensors = TransformVectorRefToMultiTensor(op_outputs);
  if (output_tensors.size() != op_outputs.size()) {
    MS_LOG(EXCEPTION) << "Op output contains tuple, node = " << kernel->DebugString();
  }
  size_t out_index = 0;
  for (const auto &output_tensor : output_tensors) {
    auto kernel_with_index = make_pair(kernel, out_index++);
    if (ref_count.find(kernel_with_index) != ref_count.end()) {
      (*op_output_map)[kernel_with_index] = output_tensor;
    }
    const auto &iter = output_indexes.find(kernel_with_index);
    if (iter == output_indexes.end()) {
      continue;
    }
    const std::vector<std::vector<size_t>> &multiple_ref_indexes = iter->second;
    for (const auto &ref_indexes : multiple_ref_indexes) {
      size_t n = 0;
      const VectorRef *cur_vector_ref = outputs;
      for (; n < ref_indexes.size() - 1; n += 1) {
        size_t index = ref_indexes.at(n);
        if (index >= cur_vector_ref->size()) {
          MS_LOG(EXCEPTION) << "Get invalid output ref index: " << index << ", size of vertor ref is "
                            << cur_vector_ref->size();
        }
        const BaseRef &base_ref = (*cur_vector_ref)[index];
        if (!utils::isa<VectorRef>(base_ref)) {
          MS_LOG(EXCEPTION) << "Get none VectorRef by ref index, index: " << index << "cur n: " << n;
        }
        cur_vector_ref = &utils::cast<VectorRef>(base_ref);
      }
      BaseRef &tensor_ref = (*const_cast<VectorRef *>(cur_vector_ref))[ref_indexes.at(n)];
      tensor_ref = output_tensor;
    }
  }
}

void GetSingleOpRunInfo(const CNodePtr cnode, OpRunInfo *run_info) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(run_info);
  auto primitive = AnfAlgo::GetCNodePrimitive(cnode);
  run_info->primitive = primitive;
  run_info->op_name = primitive->name();
  if (cnode->abstract() == nullptr) {
    MS_LOG(EXCEPTION) << "Abstract is nullptr, node = " << cnode->DebugString();
  }
  run_info->abstract = cnode->abstract();
}

GraphInfo GetSingleOpGraphInfo(const PrimitivePtr &prim, const std::vector<tensor::TensorPtr> &input_tensors) {
  MS_EXCEPTION_IF_NULL(prim);
  GraphInfo graph_info;
  // get input tensor info
  for (const auto &tensor : input_tensors) {
    MS_EXCEPTION_IF_NULL(tensor);
    auto tensor_shape = tensor->shape();
    (void)std::for_each(tensor_shape.begin(), tensor_shape.end(),
                        [&](const auto &dim) { (void)graph_info.append(std::to_string(dim) + "_"); });
    (void)graph_info.append(std::to_string(tensor->data_type()) + "_");
    if (tensor->device_address() != nullptr) {
      const auto type_id = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address())->type_id();
      (void)graph_info.append(std::to_string(type_id) + "_");
      const auto format = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address())->format();
      (void)graph_info.append(format + "_");
    }
  }
  // get attr info
  const auto &attr_map = prim->attrs();
  (void)std::for_each(attr_map.begin(), attr_map.end(),
                      [&](const auto &element) { (void)graph_info.append(element.second->ToString() + "_"); });
  const auto &added_attr_map = prim->evaluate_added_attrs();
  (void)std::for_each(added_attr_map.begin(), added_attr_map.end(),
                      [&](const auto &element) { (void)graph_info.append(element.second->ToString() + "_"); });
  graph_info.append(prim->id());
  return graph_info;
}
}  // namespace

void AscendSession::Init(uint32_t device_id) { InitExecutor(kAscendDevice, device_id); }

void AscendSession::UnifyMindIR(const KernelGraphPtr &graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    std::string file_name = "hwopt_d_before_unify_mindir_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph);
    DumpIRProto(graph, "before_unify_mindir_hwopt_" + std::to_string(graph->graph_id()));
  }
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto unify_mindir_pm = std::make_shared<opt::PassManager>("unify_mindir_pm");
  unify_mindir_pm->AddPass(std::make_shared<opt::MaxPool2MaxPoolWithArgmax>());
  unify_mindir_pm->AddPass(std::make_shared<opt::MaxPoolWithArgmaxUnifyMindIR>());
  unify_mindir_pm->AddPass(std::make_shared<opt::MaxPoolGradWithArgmaxUnifyMindIR>());
  unify_mindir_pm->AddPass(std::make_shared<opt::Conv2DUnifyMindIR>());
  unify_mindir_pm->AddPass(std::make_shared<opt::Conv2DBackpropInputUnifyMindIR>());
  unify_mindir_pm->AddPass(std::make_shared<opt::Conv2DBackpropFilterUnifyMindIR>());
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
    unify_mindir_pm->AddPass(std::make_shared<opt::DropoutGradUnifyMindIR>());
    unify_mindir_pm->AddPass(std::make_shared<opt::DropoutUnifyMindIR>());
    unify_mindir_pm->AddPass(std::make_shared<opt::SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>());
    unify_mindir_pm->AddPass(std::make_shared<opt::GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>());
    unify_mindir_pm->AddPass(std::make_shared<opt::GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2>());
  } else {
    unify_mindir_pm->AddPass(std::make_shared<opt::DropoutUnifyMindIRPynative>());
    unify_mindir_pm->AddPass(std::make_shared<opt::DropoutGradUnifyMindIRPynative>());
    unify_mindir_pm->AddPass(std::make_shared<opt::PynativeSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>());
    unify_mindir_pm->AddPass(std::make_shared<opt::PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>());
  }

  optimizer->AddPassManager(unify_mindir_pm);
  (void)optimizer->Optimize(graph);
  graph->SetExecOrderByDefault();
  if (save_graphs) {
    std::string file_name = "hwopt_d_after_unify_mindir_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph);
  }
}

GraphId AscendSession::CompileGraphImpl(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  MS_LOG(INFO) << "Start";
  // construct graph, if successfully, graph_sum_ + 1
  auto graph = ConstructKernelGraph(lst, outputs);
  auto graph_id = graph->graph_id();
  MS_LOG(INFO) << "Compile graph " << graph_id << " success";
  return graph_id;
}

bool IsBackward(const CNodePtr &cnode) {
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  return prim->HasAttr(BACKWARD);
}

// compare the value of send/recv sr_tag
bool comp(const CNodePtr &node1, const CNodePtr &node2) {
  auto prim1 = GetValueNode<PrimitivePtr>(node1->input(0));
  MS_EXCEPTION_IF_NULL(prim1);
  auto prim2 = GetValueNode<PrimitivePtr>(node1->input(0));
  MS_EXCEPTION_IF_NULL(prim2);
  auto sr_tag_value1 = prim1->GetAttr(SR_TAG);
  MS_EXCEPTION_IF_NULL(sr_tag_value1);
  auto sr_tag_value2 = prim2->GetAttr(SR_TAG);
  MS_EXCEPTION_IF_NULL(sr_tag_value2);
  auto sr_tag1 = GetValue<int64_t>(sr_tag_value1);
  auto sr_tag2 = GetValue<int64_t>(sr_tag_value2);
  return sr_tag1 < sr_tag2;
}

// Reorder the execution order of send
void ReorderSend(std::vector<CNodePtr> *execution_order, std::vector<CNodePtr> op_v) {
  auto last_node = op_v.back();
  for (auto &node : op_v) {
    if (node == last_node) {
      continue;
    }
    auto node_iter = std::find(execution_order->begin(), execution_order->end(), node);
    (void)execution_order->erase(node_iter);
  }
  std::sort(op_v.begin(), op_v.end(), comp);
  auto last_node_iter = std::find(execution_order->begin(), execution_order->end(), last_node);
  auto node_iter = execution_order->erase(last_node_iter);
  // all send will insert the end of the last node
  execution_order->insert(node_iter, op_v.begin(), op_v.end());
}

// Reorder the execution order of receive
void ReorderRecv(std::vector<CNodePtr> *execution_order, std::vector<CNodePtr> op_v) {
  auto begin_node = op_v.front();
  for (auto &node : op_v) {
    if (node == begin_node) {
      continue;
    }
    auto node_iter = std::find(execution_order->begin(), execution_order->end(), node);
    (void)execution_order->erase(node_iter);
  }
  std::sort(op_v.begin(), op_v.end(), comp);
  auto begin_node_iter = std::find(execution_order->begin(), execution_order->end(), begin_node);
  auto node_iter = execution_order->erase(begin_node_iter);
  // all receive will insert before the begin node
  execution_order->insert(node_iter, op_v.begin(), op_v.end());
}

void ReorderSendRecv(std::vector<CNodePtr> *execution_order) {
  std::vector<CNodePtr> forward_send, forward_recv, backward_send, backward_recv;
  for (auto &cnode : *execution_order) {
    if (IsPrimitiveCNode(cnode, prim::kPrimSend) && IsBackward(cnode)) {
      backward_send.push_back(cnode);
      continue;
    } else if (IsPrimitiveCNode(cnode, prim::kPrimSend)) {
      forward_send.push_back(cnode);
      continue;
    }
    if (IsPrimitiveCNode(cnode, prim::kPrimReceive) && IsBackward(cnode)) {
      backward_recv.push_back(cnode);
    } else if (IsPrimitiveCNode(cnode, prim::kPrimReceive)) {
      forward_recv.push_back(cnode);
    }
  }
  if (!forward_send.empty()) {
    ReorderSend(execution_order, forward_send);
  }
  if (!backward_send.empty()) {
    ReorderSend(execution_order, backward_send);
  }
  if (!forward_recv.empty()) {
    ReorderRecv(execution_order, forward_recv);
  }
  if (!backward_recv.empty()) {
    ReorderRecv(execution_order, backward_recv);
  }
}

GraphId AscendSession::CompileGraphImpl(NotNull<FuncGraphPtr> func_graph) {
  MS_LOG(INFO) << "Start";
  std::vector<KernelGraphPtr> all_graphs;
  auto root_graph = ConstructKernelGraph(func_graph, &all_graphs);
  // Update Graph Dynamic Shape Attr
  UpdateAllGraphDynamicShapeAttr(all_graphs);
  for (const auto &graph : all_graphs) {
    UnifyMindIR(graph);
  }
  BackendOptimization(all_graphs);
  // empty graph dont entry to backend
  if (root_graph->execution_order().empty()) {
    MS_LOG(INFO) << root_graph->ToString() << " is empty graph.";
    InsertMakeTupleForOutput(NOT_NULL(root_graph));
    root_graph->set_executable(false);
    InitRuntimeResource();
    return root_graph->graph_id();
  }

  // create parameter for multiple branch
  std::set<KernelGraphPtr> memo;
  CreateMultiBranchOutput(NOT_NULL(root_graph), NOT_NULL(&memo));
  memo.clear();
  // insert goto labels and label_sets
  LinkChildGraphs(NOT_NULL(root_graph));
  // replace labelgoto with labelswitch in subgraph called multiple times
  MultiCallGraphOptimize(NOT_NULL(root_graph));
  // resource initialize
  InitRuntimeResource();

  IrFusionPass(NOT_NULL(root_graph), NOT_NULL(&memo));
  memo.clear();

  SelectKernel(NOT_NULL(root_graph));
  memo.clear();

  HardwareOptimize(NOT_NULL(root_graph), NOT_NULL(&memo));
  memo.clear();
  // load graphs to debugger.
  if (debugger_ && debugger_->DebuggerBackendEnabled()) {
    LoadGraphsToDbg(NOT_NULL(root_graph), NOT_NULL(&memo));
  }
  memo.clear();

  UpdateRefOutputMap(NOT_NULL(root_graph), NOT_NULL(&memo));
  memo.clear();
  // add make_tuple to the output graph
  InsertMakeTupleForOutput(NOT_NULL(root_graph));
  // root root_graph valiate,include genearte execute order and so on
  RootGraphExecutorValidate(NOT_NULL(root_graph));
  // dump graph before remove nop nodes
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    DumpIRProto(root_graph, "before_removeNop_" + std::to_string(graph_sum_));
  }

  // adjust kernel
  AdjustKernel(root_graph);

  // reorder send/recv
  auto execution_order = root_graph->execution_order();
  ReorderSendRecv(&execution_order);
  root_graph->set_execution_order(execution_order);
#if ENABLE_CPU && ENABLE_D
  InitPsWorker(root_graph);
#endif
  // assign stream
  AssignStream(NOT_NULL(root_graph));
  // insert profiling point
  device::KernelAdjust::GetInstance().Profiling(NOT_NULL(root_graph.get()));
  // build kernel
  BuildKernel(root_graph);
  if (debugger_ && debugger_->partial_memory()) {
    debugger_->PreExecute(root_graph, graph_sum_);
  }
  SetSummaryNodes(root_graph.get());
  // Alloc memory for child graph's inputs
  AssignStaticMemory(NOT_NULL(root_graph), NOT_NULL(&memo));
  memo.clear();
  // Alloc memory for root graph's inputs and node's outputs, workspace
  MemoryAlloc(root_graph.get());
  // generate and load task into device
  Load(root_graph);
  root_graph->SetInputNodes();
  root_graph->SetOptimizerFlag();
  DumpAllGraphs(all_graphs);
  // return the root_graph id to backend
  auto graph_id = root_graph->graph_id();
  return graph_id;
}

void AscendSession::SetFinalGraphSummaryFlag(const std::shared_ptr<KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto graph_order = GetGraphOrder(kernel_graph->graph_id());
  for (auto graph_id : graph_order) {
    auto child_graph = GetGraph(graph_id);
    if (child_graph == nullptr) {
      continue;
    }
    if (child_graph->summary_node_exist()) {
      kernel_graph->set_summary_node_exist(true);
      return;
    }
  }
  kernel_graph->set_summary_node_exist(false);
}

void AscendSession::BuildGraphImpl(GraphId graph_id) {
  MS_LOG(INFO) << "Start";
  auto graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(graph);
  // resource initialize
  InitRuntimeResource();
  // multiple graph handle
  if (graph_id == final_graph_id_) {
    if (!graph->executable()) {
      return;
    }
    SetFinalGraphSummaryFlag(graph);
    // OptChildGraphs
    auto graph_order = GetGraphOrder(final_graph_id_);
    auto &graph_type = GetGraphOrderType(final_graph_id_);
    for (size_t i = 0; i < graph_order.size(); i++) {
      if (!(graph_type[i] == BRANCH_END || graph_type[i] == BRANCH_START)) {
        auto child_graph = GetGraph(graph_order[i]);
        CompileChildGraph(child_graph);
      }
    }
    SetSummaryNodes(graph.get());
    // merge child graph
    MergeGraphExecOrder();
  } else {
    auto single_graph = GetGraph(graph_id);
    MS_EXCEPTION_IF_NULL(single_graph);
    CompileChildGraph(single_graph);
    // set the distinction label of single graph
    single_graph->set_stream_distinction_label(graph_id);
    single_graph->UpdateExecuteKernelStreamLabel();
  }
  // adjust execution order because  merge child graph and other special operations
  AdjustKernel(graph);
#if ENABLE_CPU && ENABLE_D
  InitPsWorker(graph);
#endif
  // Reorder optimizer order
  auto execution_order = graph->execution_order();
  Reorder(&execution_order);
  graph->set_execution_order(execution_order);
  // Assign streams for control sink and hccl and so on
  AssignStream(NOT_NULL(graph));

  device::KernelAdjust::GetInstance().Profiling(NOT_NULL(graph.get()));
  // build kernel if node is cnode
  BuildKernel(graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (debugger_ && debugger_->partial_memory()) {
    debugger_->PreExecute(graph, graph_sum_);
  }
  if (ms_context->get_param<bool>(MS_CTX_PRECOMPILE_ONLY)) {
    MS_LOG(INFO) << "Precompile only, stop in build kernel step";
  } else {
    // alloc memory, including static memory and dynamic memory
    MemoryAlloc(graph.get());
    // generate and load task info to device if it is sink mode
    Load(graph);
  }
  // sync the inital const tensor to device
  SyncInitialTenosrToDevice();
  DumpAllGraphs({graph});
  MS_LOG(INFO) << "End";
}

void AscendSession::CompileChildGraph(const KernelGraphPtr &child_graph) {
  MS_EXCEPTION_IF_NULL(child_graph);
  MS_LOG(INFO) << "CompileChildGraph " << child_graph->ToString();
  opt::AscendBackendIRFusionOptimization(child_graph);
  child_graph->SetExecOrderByDefault();
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    std::string file_name = "select_kernel_before_graph_" + std::to_string(child_graph->graph_id()) + ".ir";
    DumpIR(file_name, child_graph);
  }
  // select kernel build info
  SelectKernel(*child_graph);
  if (save_graphs) {
    std::string file_name = "select_kernel_after_graph_" + std::to_string(child_graph->graph_id()) + ".ir";
    DumpIR(file_name, child_graph);
  }
  // optimize graph
  HardwareOptimize(child_graph);
  // assign static memory of parameters
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->AssignStaticMemoryInput(child_graph.get());
  runtime_instance->AssignStaticMemoryValueNode(child_graph.get());
}

bool AscendSession::IsSupportSummary() { return !device::KernelAdjust::NeedInsertSwitch(); }

void AscendSession::RunGraphImpl(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs,
                                 VectorRef *const outputs) {
  MS_LOG(INFO) << "Start";
  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // if none of child graph and no anf output exists
  if (!kernel_graph->executable()) {
    MS_LOG(INFO) << "No child graph has anf output";
    return;
  }
  // load data to extra params
  std::set<KernelGraphPtr> memo;
  SyncDataToExtraParams(NOT_NULL(kernel_graph), NOT_NULL(&memo));
  memo.clear();
  // load input data from user input
  LoadInputData(kernel_graph, inputs);
  if (debugger_) {
    debugger_->PreExecute(kernel_graph, graph_sum_);
  }
#if ENABLE_CPU && ENABLE_D
  // Initialize parameter server
  InitPSParamAndOptim(kernel_graph, inputs);
  std::string channel_name;
  if (ps::PsDataPrefetch::GetInstance().cache_enable() && IsGetNextGraph(graph_id, &channel_name)) {
    ps::ps_cache_instance.IncreaseGraphStep(channel_name);
  }
#endif
  {
    // run task on device
    Execute(kernel_graph, true);
  }
  // summary
  Summary(kernel_graph.get());
  // load tensor from device for debugger
  if (debugger_ && debugger_->debugger_enabled()) {
    LoadTensor(kernel_graph);
  }
  // debugger post-execution processing
  if (debugger_) {
    debugger_->PostExecute();
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::RunOpHardwareOptimize(const std::shared_ptr<session::KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start";
  // data layout optimization
  opt::AscendDataLayout(kernel_graph);
  // mixed precision optimization
  opt::AscendMixPrecision(kernel_graph);
  MS_LOG(INFO) << "Finish";
}

bool AscendSession::GraphCacheExist(const GraphInfo &graph_info) const {
  return run_op_graphs_.find(graph_info) != run_op_graphs_.end();
}

void AscendSession::BuildOpImpl(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                                const std::vector<tensor::TensorPtr> &input_tensors,
                                const std::vector<int64_t> &tensors_mask) {
  MS_LOG(INFO) << "Build op " << op_run_info.op_name << " start !";
  if (GraphCacheExist(graph_info)) {
    MS_LOG(INFO) << "Build op " << op_run_info.op_name << " graph cache has existed !";
    return;
  }

  // construct graph include one op
  auto graph = ConstructSingleOpGraph(op_run_info, input_tensors, tensors_mask, true);
  MS_EXCEPTION_IF_NULL(graph);
  opt::RunOpAscendBackendIRFusionOptimization(graph);
  // kernel select
  SelectKernel(*graph);
  // optimize
  RunOpHardwareOptimize(graph);
  // init runtime resource
  InitRuntimeResource();
  // build kernel
  RunOpAdjustKernel(graph);
  BuildKernel(graph);
  run_op_graphs_[graph_info] = graph;
  MS_LOG(INFO) << "Build op " << op_run_info.op_name << " finish !";
}

void AscendSession::RunOpImpl(const GraphInfo &graph_info, OpRunInfo *op_run_info,
                              std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                              const std::vector<int64_t> &tensors_mask) {
  MS_EXCEPTION_IF_NULL(input_tensors);
  MS_EXCEPTION_IF_NULL(op_run_info);
  BuildOpImpl(*op_run_info, graph_info, *input_tensors, tensors_mask);
  EraseValueNodeTensor(tensors_mask, input_tensors);
  // Run op
  auto graph = run_op_graphs_[graph_info];
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Run op " << op_run_info->op_name << " start!";
  // malloc mem
  RunOpRemoveNopNode(graph);
  RunOpMemoryAlloc(*input_tensors, graph.get());
  // Build dynamic kernel
  if (op_run_info->is_dynamic_shape) {
    BuildDynamicKernel(graph);
  }
  // load input data to device
  LoadInputData(graph, *input_tensors);
  // run op
  Execute(graph, false);
  // get output
  UpdateOutputs(graph, outputs, *input_tensors);
  // update output abstract of dynamic op to op_run_info
  if (op_run_info->is_dynamic_shape) {
    UpdateOutputAbstract(graph, op_run_info);
  }
  RunOpMemoryClear(graph.get());
  MS_LOG(INFO) << "Run op " << op_run_info->op_name << " finish!";
}

void AscendSession::RunOpsInGraphImpl(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs,
                                      VectorRef *outputs) {
  MS_LOG(INFO) << "Start!";
  auto kernel_graph = GetGraph(graph_id);
  std::map<AnfNodePtr, size_t> parameter_index;
  GetParameterIndex(kernel_graph.get(), inputs, &parameter_index);
  std::map<KernelWithIndex, std::vector<std::vector<size_t>>> output_indexes;
  CreateOutputPlaceholder(kernel_graph, inputs, outputs, &output_indexes);
  std::map<KernelWithIndex, size_t> cnode_ref;
  GetRefCount(kernel_graph.get(), &cnode_ref);

  std::map<KernelWithIndex, tensor::TensorPtr> op_output_map;
  for (const auto &kernel : kernel_graph->execution_order()) {
    // Generate input tensors, tensor masks and input kernel with index
    InputTensorInfo input_tensor_info;
    GetOpInputTensors(kernel, op_output_map, parameter_index, inputs, &input_tensor_info);

    // Get OpRunInfo and GraphInfo
    OpRunInfo run_info;
    GetSingleOpRunInfo(kernel, &run_info);
    GraphInfo graph_info = GetSingleOpGraphInfo(run_info.primitive, input_tensor_info.input_tensors);

    // Build and run current single op
    VectorRef op_outputs;
    RunOpImpl(graph_info, &run_info, &input_tensor_info.input_tensors, &op_outputs,
              input_tensor_info.input_tensors_mask);

    // Handle inputs and outputs of current op
    HandleOpInputs(input_tensor_info.input_kernel, &cnode_ref, &op_output_map);
    HandleOpOutputs(kernel, op_outputs, output_indexes, cnode_ref, &op_output_map, outputs);
  }
  MS_LOG(INFO) << "Finish!";
}

// compile graph steps
void AscendSession::SelectKernel(const KernelGraph &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  size_t raise_precision_count = 0;
  size_t reduce_precision_count = 0;
  for (const auto &cnode : kernel_graph.execution_order()) {
    auto status = device::ascend::SelectKernelInfo(cnode);
    AnfAlgo::EraseNodeAttr(kAttrPynativeNextOpName, cnode);
    AnfAlgo::EraseNodeAttr(kAttrPynativeNextIndex, cnode);
    if (status == device::ascend::kStatusRaisePrecision) {
      raise_precision_count++;
    } else if (status == device::ascend::kStatusReducePrecision) {
      reduce_precision_count++;
    }
    MS_LOG(INFO) << "Select ApplyKernel: " << cnode->DebugString();
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
    if (raise_precision_count > 0) {
      MS_LOG(WARNING) << "There has " << raise_precision_count
                      << " node/nodes used raise precision to selected the kernel!";
    }
    if (reduce_precision_count > 0) {
      MS_LOG(WARNING) << "There has " << reduce_precision_count
                      << " node/nodes used reduce precision to selected the kernel!";
    }
  }
  MS_LOG(INFO) << "Finish!";
}

void DumpInit() {
  auto &json_parser = DumpJsonParser::GetInstance();
  json_parser.Parse();
  if (json_parser.async_dump_enabled()) {
    if (AdxDataDumpServerInit() != 0) {
      MS_LOG(EXCEPTION) << "Adx data dump server init failed";
    }
  }
}

void AscendSession::InitRuntimeResource() {
  MS_LOG(INFO) << "Start!";
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  if (!runtime_instance->Init()) {
    MS_LOG(EXCEPTION) << "Kernel runtime init error.";
  }
  DumpInit();
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::HardwareOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "HardwareOptimize start!";
  opt::AscendBackendOptimization(kernel_graph);
  opt::AscendGraphKernelCommonProcess(kernel_graph);
  GraphKernelOptimize(kernel_graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  MS_LOG(INFO) << "HardwareOptimize Finish!";
}

void AscendSession::GraphKernelOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!(context_ptr->get_param<bool>(MS_CTX_ENABLE_GRAPH_KERNEL))) {
    return;
  }
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>("graph_kernel_pm");
  pm->AddPass(std::make_shared<opt::ReorderOps>());
  pm->AddPass(std::make_shared<opt::GraphKernelExpander>());
  pm->AddPass(std::make_shared<opt::BasicOpsFusion>());
  pm->AddPass(std::make_shared<opt::EliminateRedundantOutput>());
  pm->AddPass(std::make_shared<opt::GraphKernelCSE>());
  pm->AddPass(std::make_shared<opt::TensorPromotion>());
  pm->AddPass(std::make_shared<opt::GraphKernelSplitter>());
  // After Simplify and Splitter, a lot of redundant getitem/maketuple
  // will be exposed, use GetitemTuple Pass to delete them.
  pm->AddPass(std::make_shared<opt::GetitemTuple>());
  pm->AddPass(std::make_shared<opt::BindValueToGraph>());
  pm->AddPass(std::make_shared<opt::CleanAddAtomic>());
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
}

void AscendSession::AdjustKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  opt::HideNopNode(kernel_graph.get());
  // Insert CLearZero op
  // prepare for next step from json get atomic info
  BuildKernel(kernel_graph);
  device::ascend::KernelBuildPreprocess(kernel_graph.get());
  device::KernelAdjust::GetInstance().InsertSwitchLoop(kernel_graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    DumpIR("after_adjust_kernel.ir", kernel_graph);
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::RunOpAdjustKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  RunOpHideNopNode(kernel_graph);
  // Insert CLearZero op
  // prepare for next step from json get atomic info
  BuildKernel(kernel_graph);
  device::ascend::KernelBuildPreprocess(kernel_graph.get());
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::AssignStream(NotNull<KernelGraphPtr> kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  device::ascend::AscendStreamAssign::GetInstance().AssignStream(kernel_graph);
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::BuildKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  struct timeval start_time, end_time;
  (void)gettimeofday(&start_time, nullptr);
  auto ret = device::ascend::KernelBuild(kernel_graph.get());
  if (!ret) {
    MS_LOG(EXCEPTION) << "Kernel build error.";
  }
  (void)gettimeofday(&end_time, nullptr);
  const uint64_t kUSecondInSecond = 1000000;
  uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "KernelBuild run in  " << PRIu64 << " us " << cost;
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::BuildDynamicKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  MS_EXCEPTION_IF_NULL(kernel_graph);
  const auto &kernels = kernel_graph->execution_order();
  auto iter = std::find_if(kernels.begin(), kernels.end(), [](const CNodePtr &kernel) {
    return AnfAlgo::GetKernelType(kernel) == AICPU_KERNEL && AnfAlgo::GetBooleanAttr(kernel, kAttrOutputIsDynamicShape);
  });
  if (iter == kernels.end()) {
    return;
  }
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  if (!runtime_instance->GenDynamicKernel(kernel_graph.get())) {
    MS_LOG(DEBUG) << "Graph:" << kernel_graph->graph_id() << " failed to generate dynamic kernel!";
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::MemoryAlloc(KernelGraph *kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->AssignMemory(kernel_graph);
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::RunOpMemoryAlloc(const std::vector<tensor::TensorPtr> &input_tensors,
                                     KernelGraph *kernel_graph) const {
  MS_LOG(INFO) << "Start memory alloc!";
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->RunOpAssignMemory(input_tensors, kernel_graph);
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::RunOpMemoryClear(const KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->RunOpClearMemory(kernel_graph);
}

void AscendSession::Load(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool is_task_sink = context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  (void)device::KernelAdjust::GetInstance().StepLoadCtrlInputs(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  bool ret_ok = runtime_instance->Load(kernel_graph.get(), is_task_sink);
  if (!ret_ok) {
    MS_LOG(EXCEPTION) << "Load task error!";
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::Execute(const std::shared_ptr<KernelGraph> &kernel_graph, bool is_task) const {
  MS_LOG(INFO) << "Start!";
  bool is_task_sink = false;
  if (is_task) {
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    is_task_sink = context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  }
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  bool ret_ok = runtime_instance->Run(kernel_graph.get(), is_task_sink);
  Dump(kernel_graph);
  if (!ret_ok) {
    MS_LOG(EXCEPTION) << "run task error!";
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::Dump(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  MS_EXCEPTION_IF_NULL(kernel_graph);
  E2eDumpUtil::DumpData(kernel_graph.get(), device_id_);
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::DumpAllGraphs(const std::vector<KernelGraphPtr> &all_graphs) {
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (!save_graphs) {
    return;
  }
  for (auto &graph : all_graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    std::string file_name = "graph_build_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph, true);
    DumpIRProto(graph, "vm_build_" + std::to_string(graph->graph_id()));
    DumpIR("trace_code_graph", graph, true, kWholeStack);
  }
#endif
}

void AscendSession::LoadTensor(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  (void)runtime_instance->LoadData(kernel_graph.get());
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::RecurseSetSummaryNodes(KernelGraph *graph,
                                           std::map<std::string, std::pair<AnfNodePtr, int>> *summary) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(summary);
  // if final graph have no child graph
  auto graph_order_iter = graph_execute_orders_.find(graph->graph_id());
  if (graph_order_iter == graph_execute_orders_.end()) {
    SessionBasic::SetSummaryNodes(graph);
    auto summary_nodes = graph->summary_nodes();
    summary->insert(summary_nodes.begin(), summary_nodes.end());
    return;
  }
  // for every child graph, find summary nodes
  auto graph_order = GetGraphOrder(graph->graph_id());
  for (size_t i = 0; i < graph_order.size(); i++) {
    auto child_graph = GetGraph(graph_order[i]);
    if (child_graph == nullptr) {
      continue;
    }
    SessionBasic::SetSummaryNodes(child_graph.get());
    auto child_graph_summary = child_graph->summary_nodes();
    summary->insert(child_graph_summary.begin(), child_graph_summary.end());
    RecurseSetSummaryNodes(child_graph.get(), summary);
  }
  graph->set_summary_nodes(*summary);
}

void AscendSession::SetSummaryNodes(KernelGraph *graph) {
  MS_LOG(DEBUG) << "Update summary Start";
  MS_EXCEPTION_IF_NULL(graph);
  auto summary_nodes = graph->summary_nodes();
  std::map<std::string, std::pair<AnfNodePtr, int>> summary;
  summary.insert(summary_nodes.begin(), summary_nodes.end());
  RecurseSetSummaryNodes(graph, &summary);
  graph->set_summary_nodes(summary);
  MS_LOG(DEBUG) << "Update summary end size: " << summary.size();
}

void AscendSession::MergeGraphExecOrder() {
  MS_LOG(INFO) << "Start!";
  // merge graph order
  auto &graph_order = GetGraphOrder(final_graph_id_);
  auto &graph_type = GetGraphOrderType(final_graph_id_);
  auto final_graph = GetGraph(final_graph_id_);
  MS_EXCEPTION_IF_NULL(final_graph);
  if (graph_order.empty()) {
    MS_LOG(WARNING) << "Graph output is a lonely variable not linked to any op!";
    return;
  }
  if (graph_order.size() > 1) {
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    if (!context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK)) {
      MS_LOG(EXCEPTION) << "Control sink network should run with task-sink mode!";
    }
  }
  // if first graph is common,the final graph has no label,then set the stream of final graph same with the first graph
  SetStreamDistinctionLabel(final_graph, graph_order[0], false);
  std::vector<CNodePtr> final_exec_order = final_graph->execution_order();
  KernelGraphPtr last_graph = nullptr;
  for (size_t i = 0; i < graph_order.size(); i++) {
    auto graph_id = graph_order[i];
    if (graph_type[i] == BRANCH_END || graph_type[i] == BRANCH_START) {
      continue;
    }
    auto child_graph = GetGraph(graph_id);
    last_graph = child_graph;
    MS_EXCEPTION_IF_NULL(child_graph);
    auto exec_order = child_graph->execution_order();
    MS_LOG(INFO) << "Merge graph,graph_id " << graph_id;
    (void)std::transform(exec_order.begin(), exec_order.end(), std::back_inserter(final_exec_order),
                         [&](CNodePtr node) -> CNodePtr {
                           AnfAlgo::SetStreamDistinctionLabel(child_graph->stream_distinction_label(), node.get());
                           return node;
                         });
    // add all value nodes of child graphs to final graph
    for (auto &value_node : child_graph->graph_value_nodes()) {
      final_graph->AddValueNodeToGraph(value_node);
    }
    // copy ref map to final graph
    auto child_ref_map = child_graph->GetRefMap();
    for (auto &item : child_ref_map) {
      if (final_graph->IsInRefOutputMap(item.first)) {
        MS_LOG(EXCEPTION) << "The ref pair is already in final graph!";
      }
      final_graph->AddRefCorrespondPairs(item.first, item.second);
    }
  }
  // set final_exec_order into final graph
  MS_EXCEPTION_IF_NULL(final_graph);
  DumpGraphExeOrder(final_exec_order);
  final_graph->set_execution_order(final_exec_order);
}

const std::vector<GraphId> &AscendSession::GetGraphOrder(GraphId final_graph_id) const {
  auto graph_order_iter = graph_execute_orders_.find(final_graph_id);
  if (graph_order_iter == graph_execute_orders_.end()) {
    MS_LOG(EXCEPTION) << "Final graph" << final_graph_id << "has no child graph";
  }
  return graph_order_iter->second;
}

const std::vector<GraphType> &AscendSession::GetGraphOrderType(GraphId final_graph_id) const {
  auto graph_type_iter = graph_order_types_.find(final_graph_id);
  if (graph_type_iter == graph_order_types_.end()) {
    MS_LOG(EXCEPTION) << "Final graph" << final_graph_id << "has no graph_order_types_";
  }
  return graph_type_iter->second;
}

void AscendSession::SyncInitialTenosrToDevice() {
  for (auto &item : initial_tenosrs_) {
    auto to_graph_id = item.first.first;
    auto input_idx = item.first.second;
    auto front_tensor = item.second;
    auto to_graph = GetGraph(to_graph_id);
    MS_EXCEPTION_IF_NULL(to_graph);
    std::vector<AnfNodePtr> graph_inputs = to_graph->inputs();
    if (input_idx >= graph_inputs.size()) {
      MS_LOG(EXCEPTION) << "Input_index " << input_idx << " out of range size " << graph_inputs.size();
    }
    auto backend_parameter = graph_inputs[input_idx];
    // sync data from host to device
    MS_EXCEPTION_IF_NULL(front_tensor);
    size_t tensor_size = front_tensor->data().nbytes();
    auto addr = AnfAlgo::GetOutputAddr(backend_parameter, 0);
    MS_EXCEPTION_IF_NULL(addr);
    if (!addr->SyncHostToDevice(trans::GetRuntimePaddingShape(backend_parameter, 0), tensor_size,
                                front_tensor->data_type(), front_tensor->data_c())) {
      MS_LOG(EXCEPTION) << "Tensor SyncHostToDevice fail!";
    }
  }
}

void AscendSession::BackendOptimization(const std::vector<KernelGraphPtr> &all_graphs) {
  MS_LOG(INFO) << "Start BackendCommonOptimization";
  for (auto &graph : all_graphs) {
    opt::BackendCommonOptimization(graph);
  }
  MS_LOG(INFO) << "End.";
}

void AscendSession::LinkChildGraphs(NotNull<KernelGraphPtr> graph) { AscendControlParser::LinkGraph(graph); }

bool AscendSession::IsMultiCallGraph(NotNull<KernelGraphPtr> graph, std::vector<GraphId> parent_graphs) {
  std::stack<GraphId> post_graph;
  std::set<GraphId> memo;
  post_graph.push(graph->graph_id());
  while (!post_graph.empty()) {
    auto graph_id = post_graph.top();
    post_graph.pop();
    memo.insert(graph_id);
    for (auto child_graph : graphs_[graph_id]->child_graph_order()) {
      std::shared_ptr<KernelGraph> child_graph_ptr = child_graph.lock();
      MS_EXCEPTION_IF_NULL(child_graph_ptr);
      if (std::find(parent_graphs.begin(), parent_graphs.end(), child_graph_ptr->graph_id()) != parent_graphs.end()) {
        MS_LOG(DEBUG) << "graph:" << graph->graph_id() << " will call its parent graph:" << child_graph_ptr->graph_id();
        return false;
      } else if (memo.find(child_graph_ptr->graph_id()) == memo.end()) {
        MS_LOG(DEBUG) << "child graph:" << child_graph_ptr->graph_id() << " into deque, wait for check.";
        post_graph.push(child_graph_ptr->graph_id());
      }
    }
  }
  return true;
}

void AscendSession::MultiCallGraphOptimize(NotNull<KernelGraphPtr> root_graph) {
  for (auto current : parent_graphs_) {
    if (current.second.size() < 2) {
      continue;
    }
    auto graph = graphs_[current.first];
    auto parent_kernel_graphs = current.second;
    if (!IsMultiCallGraph(NOT_NULL(graph), parent_kernel_graphs)) {
      MS_LOG(DEBUG) << "graph:" << graph->graph_id() << " with it's parent graphs make up a cycle";
      continue;
    }
    MS_LOG(INFO) << "graph: " << graph->graph_id() << " has been called by more than two graphs";
    int32_t index = 0;
    std::vector<KernelGraphPtr> child_graphs;
    auto start_label_id = AnfAlgo::GetNodeAttr<uint32_t>(graph->get_start_label(), kAttrLabelIndex);
    auto end_node = graph->get_end_goto();
    ParameterPtr post_label_param = graph->AddExtraParamAndTensor("label_param", 0);
    std::vector<AnfNodePtr> new_inputs = {std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSwitchOpName)),
                                          post_label_param};
    for (auto graph_id : parent_kernel_graphs) {
      auto kg = graphs_[graph_id];
      auto nodes = kg->execution_order();
      for (uint32_t i = 0; i < nodes.size(); i++) {
        if (AnfAlgo::IsLabelIndexInNode(nodes[i], start_label_id)) {
          if (i < (nodes.size() - 1)) {
            new_inputs.push_back(nodes[i + 1]);
          } else {
            MS_LOG(EXCEPTION) << "No labelset after labelgoto";
          }
          ParameterPtr pre_label_param = kg->AddExtraParamAndTensor("label_param", index++);
          AscendControlParser::InsertMultipleAssignToGraph(NOT_NULL(kg), nodes[i], NOT_NULL(pre_label_param),
                                                           NOT_NULL(post_label_param));
        }
      }
      kg->SetExecOrderByDefault();
      child_graphs.push_back(kg);
    }
    end_node->set_inputs(new_inputs);
    AnfAlgo::SetNodeAttr(kAttrChildGraph, MakeValue<std::vector<KernelGraphPtr>>(child_graphs), end_node);
    std::vector<uint32_t> label_list;
    for (size_t i = kLabelSwitchLabelId; i < end_node->size(); ++i) {
      auto input = end_node->input(i);
      MS_EXCEPTION_IF_NULL(input);
      if (!input->isa<CNode>() || AnfAlgo::GetCNodeName(input) != kLabelSetOpName) {
        break;
      }
      uint32_t goto_label_id = AnfAlgo::GetNodeAttr<uint32_t>(input, kAttrLabelIndex);
      label_list.push_back(goto_label_id);
      MS_LOG(INFO) << "Switch " << end_node->DebugString() << " case " << i - kLabelSwitchLabelId << ": id "
                   << goto_label_id;
    }
    AnfAlgo::SetNodeAttr(kAttrLabelSwitchList, MakeValue<std::vector<uint32_t>>(label_list), end_node);
    end_node->set_inputs({end_node->input(kAnfPrimitiveIndex), end_node->input(kFirstDataInputIndex)});
    graph->SetExecOrderByDefault();
  }
}

void AscendSession::SyncDataToExtraParams(NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo) {
  if (memo->find(graph.get()) != memo->end()) {
    return;
  }
  memo->insert(graph.get());
  auto extra_param_tensor = graph->GetExtraParamAndTensor();
  for (uint32_t i = 0; i < extra_param_tensor.size(); i++) {
    auto param = extra_param_tensor[i].first;
    auto tensor = extra_param_tensor[i].second;
    auto device_address = AnfAlgo::GetMutableOutputAddr(param, 0);
    MS_EXCEPTION_IF_NULL(device_address);
    tensor->set_device_address(device_address);
    if (!device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(param, 0), LongToSize(tensor->data().nbytes()),
                                          tensor->data_type(), tensor->data_c())) {
      MS_LOG(EXCEPTION) << "SyncHostToDevice failed.";
    }
  }
  for (auto &child_graph : graph->child_graph_order()) {
    SyncDataToExtraParams(NOT_NULL(child_graph.lock()), memo);
  }
}

void AscendSession::RootGraphExecutorValidate(NotNull<KernelGraphPtr> graph) {
  AscendControlParser::ExecutorValidate(graph);
}

void AscendSession::CreateMultiBranchOutput(NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo) {
  if (memo->find(graph.get()) != memo->end()) {
    return;
  }
  memo->insert(graph.get());
  graph->UpdateChildGraphOrder();
  for (auto &child_graph : graph->child_graph_order()) {
    CreateMultiBranchOutput(NOT_NULL(child_graph.lock()), memo);
  }
  std::map<AnfNodePtr, AnfNodePtr> need_replace_list;
  auto node_list = GetCNodes(TopoSort(graph->get_return()));
  for (auto &node : node_list) {
    if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimCall) || AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitch)) {
      // create a parameter to store the output of multiple branch and set the parameter as the condition graph's output
      auto output_param = graph->TransTupleToMakeTuple(graph->NewParameter(node->abstract()));
      MS_EXCEPTION_IF_NULL(graph->MutableInputs());
      graph->AddChildGraphResult(output_param);

      std::vector<AnfNodePtr> depend_inputs = {
        graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name()))), output_param, node};
      auto depend = graph->NewCNode(depend_inputs);
      depend->set_abstract(output_param->abstract());
      need_replace_list.emplace(node, depend);
      MS_LOG(INFO) << "Create parameter " << output_param->DebugString() << " for call node " << node->DebugString()
                   << ", depend node is " << depend->DebugString();
      // insert assign in order to transfer child graph output to parameter
      auto child_graphs = AnfAlgo::GetCallSwitchKernelGraph(node);
      for (auto &child_graph : child_graphs) {
        MS_EXCEPTION_IF_NULL(child_graph);
        // If graph has no output, the graph is the true graph of while and will call condition graph, no need insert
        // assign from condition to true graph
        if (memo->find(child_graph) != memo->end()) {
          continue;
        }
        AscendControlParser::InsertMultipleAssignToGraph(NOT_NULL(child_graph), nullptr,
                                                         NOT_NULL(child_graph->output()), NOT_NULL(output_param));
      }
    }
  }
  // searching for nodes' input to replace call by depend(parameter, call)
  for (auto &node : node_list) {
    for (size_t i = 0; i < node->size(); ++i) {
      auto input = node->input(i);
      auto iter = need_replace_list.find(input);
      if (iter != need_replace_list.end()) {
        node->set_input(i, iter->second);
      }
    }
  }
  memo->erase(graph.get());
}

void AscendSession::IrFusionPass(const NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo) {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph.get());
  opt::AscendBackendIRFusionOptimization(graph);
  graph->SetExecOrderByDefault();

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    std::string file_name = "select_kernel_before_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph.get());
  }

  for (auto &child_graph : graph->child_graph_order()) {
    IrFusionPass(NOT_NULL(child_graph.lock()), memo);
  }
}

void AscendSession::SelectKernel(NotNull<KernelGraphPtr> root_graph) {
  MS_LOG(INFO) << "Start select kernel.";
  size_t raise_precision_count = 0;
  size_t reduce_precision_count = 0;

  std::set<KernelGraphPtr> memo;
  (void)RecurseSelectKernelInfo(root_graph, NOT_NULL(&memo), &raise_precision_count, &reduce_precision_count);
  memo.clear();

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
    if (raise_precision_count > 0) {
      MS_LOG(WARNING) << "There are " << raise_precision_count
                      << " node/nodes used raise precision to selected the kernel!";
    }
    if (reduce_precision_count > 0) {
      MS_LOG(WARNING) << "There are " << reduce_precision_count
                      << " node/nodes used reduce precision to selected the kernel!";
    }
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::RecurseSelectKernelInfo(NotNull<KernelGraphPtr> graph,
                                            NotNull<std::set<KernelGraphPtr> *> const memo,
                                            size_t *const raise_precision_count,
                                            size_t *const reduce_precision_count) const {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph.get());
  MS_LOG(INFO) << "Start to select kernel info in graph: " << graph->graph_id();

  for (const auto &cnode : graph->execution_order()) {
    if (AnfAlgo::IsCondControlKernel(cnode)) {
      std::vector<KernelGraphPtr> child_graphs;
      if (AnfAlgo::HasNodeAttr(kAttrChildGraph, cnode)) {
        child_graphs = AnfAlgo::GetNodeAttr<std::vector<KernelGraphPtr>>(cnode, kAttrChildGraph);
      }
      for (auto &child_graph : child_graphs) {
        RecurseSelectKernelInfo(NOT_NULL(child_graph), memo, raise_precision_count, reduce_precision_count);
      }
    }

    auto status = device::ascend::SelectKernelInfo(cnode);
    if (status == device::ascend::kStatusRaisePrecision) {
      (*raise_precision_count)++;
    } else if (status == device::ascend::kStatusReducePrecision) {
      (*reduce_precision_count)++;
    }
    MS_LOG(INFO) << "Select ApplyKernel: " << cnode->DebugString();
  }

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    std::string file_name = "select_kernel_after_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph.get());
  }
  MS_LOG(INFO) << "Finish selecting kernel info in graph: " << graph->graph_id();
}

void AscendSession::HardwareOptimize(NotNull<KernelGraphPtr> graph,
                                     NotNull<std::set<KernelGraphPtr> *> const memo) const {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph.get());

  MS_LOG(INFO) << "Start to do HardwareOptimize in graph: " << graph->graph_id();

  HardwareOptimize(graph.get());
  for (auto &child_graph : graph->child_graph_order()) {
    HardwareOptimize(NOT_NULL(child_graph.lock()), memo);
  }
  MS_LOG(INFO) << "Finish doing HardwareOptimize in graph: " << graph->graph_id();
}

void AscendSession::LoadGraphsToDbg(NotNull<KernelGraphPtr> graph,
                                    NotNull<std::set<KernelGraphPtr> *> const memo) const {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph.get());

  MS_LOG(INFO) << "Start to do LoadGraphsToDbg in graph: " << graph->graph_id();

  debugger_->LoadGraphs(graph);
  MS_LOG(INFO) << "graph_sum_: " << graph_sum_;
  for (auto &child_graph : graph->child_graph_order()) {
    LoadGraphsToDbg(NOT_NULL(child_graph.lock()), memo);
  }
  MS_LOG(INFO) << "Finish doing LoadGraphsToDbg in graph: " << graph->graph_id();
}

void AscendSession::AssignStaticMemory(NotNull<KernelGraphPtr> graph,
                                       NotNull<std::set<KernelGraphPtr> *> const memo) const {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph.get());

  MS_LOG(INFO) << "Start to assign static memory for parameter in graph: " << graph->graph_id();
  // assign static memory for parameters
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->ClearGlobalIdleMem();
  runtime_instance->AssignStaticMemoryInput(graph.get().get());
  runtime_instance->AssignStaticMemoryValueNode(graph.get().get());
  for (auto &child_graph : graph->child_graph_order()) {
    AssignStaticMemory(NOT_NULL(child_graph.lock()), memo);
  }
  MS_LOG(INFO) << "Finish assigning static memory for parameter in graph: " << graph->graph_id();
}

void AscendSession::UpdateRefOutputMap(NotNull<KernelGraphPtr> graph,
                                       NotNull<std::set<KernelGraphPtr> *> const memo) const {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph.get());

  for (auto &child_graph : graph->child_graph_order()) {
    std::shared_ptr<KernelGraph> child_graph_ptr = child_graph.lock();
    MS_EXCEPTION_IF_NULL(child_graph_ptr);
    UpdateRefOutputMap(NOT_NULL(child_graph_ptr), memo);
    // copy ref map to final graph
    auto child_ref_map = child_graph_ptr->GetRefMap();
    for (auto &item : child_ref_map) {
      if (graph->IsInRefOutputMap(item.first)) {
        MS_LOG(WARNING) << "The ref pair <" << item.first.first->DebugString() << ", " << item.first.second
                        << "> is already in " << graph->ToString();
        continue;
      }
      graph->AddRefCorrespondPairs(item.first, item.second);
    }
  }
}

GraphId AscendSession::CompileGraphImpl(NotNull<FuncGraphPtr> func_graph, const vector<tensor::TensorPtr> &inputs) {
  RunInfer(func_graph, inputs);
  return CompileGraphImpl(func_graph);
}

void AscendSession::SyncStream() {
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  auto ret = runtime_instance->SyncStream();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Sync stream error!";
  }
}
}  // namespace session
}  // namespace mindspore
