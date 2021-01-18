/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "backend/optimizer/somas/somas.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>

#include "backend/optimizer/somas/somas_node.h"
#include "backend/optimizer/somas/somas_solver_pre.h"
#include "backend/optimizer/somas/somas_stream.h"
#include "backend/optimizer/somas/somas_tensor.h"
#ifdef ENABLE_D
#include "runtime/device/ascend/ascend_stream_assign.h"
#endif
#include "backend/optimizer/common/helper.h"
#include "utils/ms_context.h"
#include "debug/common.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace somas {
constexpr auto kGapSize = 512;
constexpr auto kParallelComputeSizeThreshold = 2000;
std::map<TensorType, std::string> tensor_type_name_map = {{kCommon, "Common"},
                                                          {kOutputOnly, "OutputOnly"},
                                                          {kWorkspace, "Workspace"},
                                                          {kGetNextOutput, "GetNextOutput"},
                                                          {kSummaryInput, "SummaryInput"},
                                                          {kRefNodeInput, "RefNodeInput"},
                                                          {kRefNodeOutput, "RefNodeOutput"},
                                                          {kUnknown, "Unknown"}};

bool Somas::Allocate(const session::KernelGraph *graph) {
  auto ret = InitSomasTensors(graph);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Somas Initialize Failed.";
  }

  // Computing Conflict pairs
  MS_LOG(INFO) << "Start Computing Conflict Pairs";
  ComputeConflictPairs();
  MS_LOG(INFO) << "End Computing Conflict Pairs";

  ret = Assign(graph);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Somas Assign Failed.";
  }

  GenGraphStatisticInfo();
  return ret;
}

bool Somas::InitSomasTensors(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  InitBasicInfo(graph);
  IndependentNodeOutputProcess(graph);
  SummaryInputProcess(graph);
  RefNodeProcess(graph);
  UnReuseNodeProcess(graph);
  GenContiguousList(graph);
  GetNextOutputProcess(graph);

  if (tensors_list_.empty()) {
    MS_LOG(INFO) << "No Tensor from graph " << graph->graph_id();
    return true;
  }

  MS_LOG(INFO) << "Created " << streams_list_.size() << " streams (" << streams_groups_.size() << " groups), "
               << nodes_list_.size() << " nodes, " << tensors_list_.size() << " tensors, and "
               << contiguous_tensors_list_.size() << " contiguous lists";

  if (save_graphs_) {
    std::string file_path =
      save_graphs_path_ + "/" + "somas_pre_processed_info_" + std::to_string(graph->graph_id()) + ".ir";
    DumpSomasInfoIR(file_path);

    std::string offline_file_path =
      save_graphs_path_ + "/" + "somas_offline_log_" + std::to_string(graph->graph_id()) + ".ir";
    DumpOfflineIR(offline_file_path);
  }

  return true;
}

void Somas::InitSomasStreamAndNode(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  streams_list_ = {};
  nodes_list_ = {};
  size_t node_index = 0;
  auto kernel_cnodes = graph->execution_order();
  for (const auto &kernel : kernel_cnodes) {
    SomasStreamPtr stream;
    auto stream_id = AnfAlgo::GetStreamId(kernel);
    auto it = find_if(streams_list_.begin(), streams_list_.end(),
                      [stream_id](const SomasStreamPtr &s) { return s->GetId() == stream_id; });
    if (it == streams_list_.end()) {
      stream = std::make_shared<SomasStream>(stream_id);
      streams_list_.push_back(stream);
    } else {
      stream = *it;
    }

    // Node
    NodeType type = kCommonNode;
    if (AnfAlgo::IsCommunicationOp(kernel)) {
      type = kCommunicationNode;
    }
    auto node = std::make_shared<SomasNode>(node_index, type, stream);
    MS_EXCEPTION_IF_NULL(node);
    node->scope_full_name_ = kernel->fullname_with_scope();
    nodes_list_.push_back(node);
    stream->nodes_.push_back(node);
    auto key = kernel.get();
    nodes_map_[key] = node;
    node_index++;
  }
}

void Somas::InitSomasOutputAndWorkspaceTensors(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  tensors_list_ = {};
  size_t tensor_index = 0;
  auto kernel_cnodes = graph->execution_order();
  for (const auto &kernel : kernel_cnodes) {
    auto node = nodes_map_[kernel.get()];
    MS_EXCEPTION_IF_NULL(node);
    auto stream = node->GetStream();
    MS_EXCEPTION_IF_NULL(stream);

    // Output Tensor
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto output_sizes = kernel_mod->GetOutputSizeList();
    for (const auto &size : output_sizes) {
      auto output_tensor_index = tensor_index;
      tensor_index++;
      // Set all output tensor lifelong to true.
      auto tensor = std::make_shared<SomasTensor>(output_tensor_index, node, stream, size, kLifeLongNone);
      tensor->lifetime_.start_ = node->GetId();
      tensor->lifetime_.end_ = node->GetId();
      tensor->type_ = kOutputOnly;
      tensors_list_.push_back(tensor);
      tensors_map_[output_tensor_index] = tensor;
      stream->tensors_.push_back(tensor);
      node->tensors_.insert(tensor);
      node->output_tensors_.push_back(tensor);
    }

    // WorkSpace Tensor
    auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();
    for (const auto &size : workspace_sizes) {
      auto workspace_tensor_index = tensor_index;
      tensor_index++;
      SomasTensorPtr tensor = std::make_shared<SomasTensor>(workspace_tensor_index, node, stream, size, kLifeLongNone);
      tensor->type_ = kWorkspace;
      tensor->lifetime_.start_ = node->GetId();
      tensor->lifetime_.end_ = node->GetId();
      tensors_list_.push_back(tensor);
      tensors_map_[workspace_tensor_index] = tensor;
      stream->tensors_.push_back(tensor);
      node->tensors_.insert(tensor);
      node->workspace_tensors_.push_back(tensor);
    }
  }
}

void Somas::InitSomasInputTensors(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  bool is_all_nop_node = opt::IsAllNopNode(graph);
  auto kernel_cnodes = graph->execution_order();
  for (const auto &kernel : kernel_cnodes) {
    if (AnfAlgo::GetCNodeName(kernel) != kAtomicAddrCleanOpName) {
      InitCommonNodeInputs(is_all_nop_node, kernel);
    } else {
      InitAtomicCleanInputs(is_all_nop_node, kernel);
    }
  }
}
void Somas::InitCommonNodeInputs(bool is_all_nop_node, const CNodePtr &kernel) {
  auto node = nodes_map_[kernel.get()];
  MS_EXCEPTION_IF_NULL(node);
  auto stream = node->GetStream();
  MS_EXCEPTION_IF_NULL(stream);

  // Input Tensor
  auto input_tensor_num = AnfAlgo::GetInputTensorNum(kernel);
  for (size_t i = 0; i < input_tensor_num; i++) {
    auto input_node = kernel->input(i + 1);
    session::KernelWithIndex prenode_index;
    if (is_all_nop_node) {
      prenode_index = AnfAlgo::VisitKernelWithReturnType(input_node, 0, false);
    } else {
      prenode_index = AnfAlgo::VisitKernelWithReturnType(input_node, 0, true);
    }
    if (AnfAlgo::CheckPrimitiveType(prenode_index.first, prim::kPrimMakeTuple)) {
      MS_LOG(EXCEPTION) << "Input node [" << input_node->DebugString() << "]'s input " << i << " is MakeTuple";
    }

    if (!AnfAlgo::IsRealCNodeKernel(prenode_index.first)) {
      auto op_name = AnfAlgo::GetCNodeName(kernel);
      TypeId input_origin_type = AnfAlgo::GetPrevNodeOutputInferDataType(kernel, i);
      if ((op_name == kDynamicRNNOpName || op_name == kDynamicGRUV2OpName) && input_origin_type == kMetaTypeNone) {
        continue;
      }
      auto parameter = GetSomasParameters(prenode_index.first, prenode_index.second);
      node->input_parameters_map_[i] = parameter;
      MS_LOG(DEBUG) << "Input  [" << prenode_index.first->fullname_with_scope() << "] is not a real cnode kernel.";
      continue;
    }

    auto iter = nodes_map_.find(prenode_index.first.get());
    if (iter == nodes_map_.end()) {
      MS_LOG(EXCEPTION) << "Kernel[" << kernel->fullname_with_scope() << "]'s input " << i << " ["
                        << prenode_index.first->fullname_with_scope() << "] is not init.";
    }
    auto pre_somas_node = iter->second;
    if (prenode_index.second > pre_somas_node->output_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Output index " << prenode_index.second << " exceed input node ["
                        << prenode_index.first->fullname_with_scope() << "]'s outputs size "
                        << pre_somas_node->output_tensors_.size();
    }
    auto input_somas_tensor = pre_somas_node->output_tensors_[prenode_index.second];
    MS_EXCEPTION_IF_NULL(input_somas_tensor);
    node->input_tensors_.push_back(input_somas_tensor);
    if (input_somas_tensor->type_ == kOutputOnly) {
      input_somas_tensor->type_ = kCommon;
    }
    input_somas_tensor->destinations_.insert(node);
    input_somas_tensor->destinationStreams_.insert(stream);
    if (input_somas_tensor->lifetime_.end_ < node->GetId()) {
      input_somas_tensor->lifetime_.end_ = node->GetId();
    }

    if (node != pre_somas_node) {
      node->ancestor_nodes_.insert(pre_somas_node);
    }
    auto input_tensor_stream = input_somas_tensor->GetSourceStream();
    if (input_tensor_stream != stream) {
      stream->ancestor_streams_.insert(input_tensor_stream);
      input_somas_tensor->between_streams_ = true;
    }
  }
}

void Somas::InitAtomicCleanInputs(bool is_all_nop_node, const CNodePtr &kernel) {
  auto node = nodes_map_[kernel.get()];
  MS_EXCEPTION_IF_NULL(node);
  auto stream = node->GetStream();
  MS_EXCEPTION_IF_NULL(stream);

  MS_EXCEPTION_IF_NULL(kernel->inputs()[1]);
  auto pre_node = (kernel->inputs()[1])->cast<CNodePtr>();
  auto iter = nodes_map_.find(pre_node.get());
  if (iter == nodes_map_.end()) {
    MS_LOG(EXCEPTION) << "Kernel[" << kernel->fullname_with_scope() << "]'s input [" << pre_node->fullname_with_scope()
                      << "] is not init.";
  }
  auto pre_somas_node = iter->second;
  // set clean output tensors
  if (AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, pre_node)) {
    auto clean_output_indexs = AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicOutputIndexs);
    for (auto index : clean_output_indexs) {
      if (index > pre_somas_node->output_tensors_.size()) {
        MS_LOG(EXCEPTION) << "Output index " << index << " exceed input node [" << pre_node->fullname_with_scope()
                          << "]'s outputs size " << pre_somas_node->output_tensors_.size();
      }
      auto input_somas_tensor = pre_somas_node->output_tensors_[index];
      MS_EXCEPTION_IF_NULL(input_somas_tensor);
      node->input_tensors_.push_back(input_somas_tensor);
      input_somas_tensor->destinations_.insert(node);
      input_somas_tensor->destinationStreams_.insert(stream);
      if (input_somas_tensor->lifetime_.start_ > node->GetId()) {
        input_somas_tensor->lifetime_.start_ = node->GetId();
      }
      node->ancestor_nodes_.insert(pre_somas_node);
      auto input_tensor_stream = input_somas_tensor->GetSourceStream();
      if (input_tensor_stream != stream) {
        stream->ancestor_streams_.insert(input_tensor_stream);
        input_somas_tensor->between_streams_ = true;
      }
    }
  }
  // set clean workspace tensors
  if (AnfAlgo::HasNodeAttr(kAttrAtomicWorkspaceIndexs, pre_node)) {
    auto clean_workspace_indexs = AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicWorkspaceIndexs);
    for (const auto &index : clean_workspace_indexs) {
      if (index > pre_somas_node->output_tensors_.size()) {
        MS_LOG(EXCEPTION) << "Workspace index " << index << " exceed input node [" << pre_node->fullname_with_scope()
                          << "]'s Workspace size " << pre_somas_node->workspace_tensors_.size();
      }
      auto input_somas_tensor = pre_somas_node->workspace_tensors_[index];
      MS_EXCEPTION_IF_NULL(input_somas_tensor);
      node->input_tensors_.push_back(input_somas_tensor);
      input_somas_tensor->destinations_.insert(node);
      input_somas_tensor->destinationStreams_.insert(stream);
      if (input_somas_tensor->lifetime_.start_ > node->GetId()) {
        input_somas_tensor->lifetime_.start_ = node->GetId();
      }
      node->ancestor_nodes_.insert(pre_somas_node);
      auto input_tensor_stream = input_somas_tensor->GetSourceStream();
      if (input_tensor_stream != stream) {
        stream->ancestor_streams_.insert(input_tensor_stream);
        input_somas_tensor->between_streams_ = true;
      }
    }
  }
}

SomasParameterPtr Somas::CreateSomasParameters(AnfNodePtr node, size_t index) {
  auto id = parameters_list_.size();
  auto device_addr = AnfAlgo::GetOutputAddr(node, index);
  if (device_addr == nullptr) {
    MS_LOG(EXCEPTION) << "Node " << node->fullname_with_scope() << " has no device address before Somas.";
  }
  auto param = std::make_shared<SomasParameter>(id, node, index, device_addr->GetPtr(), device_addr->GetSize());
  parameters_list_.push_back(param);
  return param;
}

SomasParameterPtr Somas::GetSomasParameters(AnfNodePtr node, size_t index) {
  auto key = node.get();
  auto iter = parameters_map_.find(key);
  if (iter != parameters_map_.end()) {
    auto it = std::find_if(iter->second.begin(), iter->second.end(),
                           [index](SomasParameterPtr param) -> bool { return index == param->output_index_; });
    if (it != iter->second.end()) {
      return *it;
    } else {
      auto new_param = CreateSomasParameters(node, index);
      iter->second.push_back(new_param);
      return new_param;
    }
  } else {
    auto new_param = CreateSomasParameters(node, index);
    parameters_map_[key].push_back(new_param);
    return new_param;
  }
}

void Somas::InitBasicInfo(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
#ifdef ENABLE_D
  streams_groups_ = device::ascend::AscendStreamAssign::GetInstance().get_stream_group();
#endif
  InitSomasStreamAndNode(graph);
  InitSomasOutputAndWorkspaceTensors(graph);
  InitSomasInputTensors(graph);

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  save_graphs_ = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  save_graphs_path_ = context_ptr->get_param<std::string>(MS_CTX_SAVE_GRAPHS_PATH);
  if (save_graphs_path_.empty()) {
    save_graphs_path_ = ".";
  }
  if (save_graphs_) {
    std::string file_path = save_graphs_path_ + "/" + "somas_initial_info_" + std::to_string(graph->graph_id()) + ".ir";
    DumpSomasInfoIR(file_path);
  }
}

void Somas::GetNextOutputProcess(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_cnodes = graph->execution_order();
  size_t total_size = 0;
  for (const auto &kernel : kernel_cnodes) {
    if (AnfAlgo::GetCNodeName(kernel) != kGetNextOpName) {
      continue;
    }
    auto iter = nodes_map_.find(kernel.get());
    if (iter != nodes_map_.end()) {
      auto getnext_output_tensors = iter->second->output_tensors_;
      for (auto &tensor : getnext_output_tensors) {
        total_size += tensor->GetAlignedSize();
        tensor->lifelong_value_ = kLifeLongGraphAll;
        tensor->type_ = kGetNextOutput;
      }
    }
  }
  MS_LOG(INFO) << "Special Tensor total size: GetNext Output " << total_size;
}

void Somas::IndependentNodeOutputProcess(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_cnodes = graph->execution_order();
  size_t total_size = 0;
  for (const auto &kernel : kernel_cnodes) {
    bool independent = AnfAlgo::IsIndependentNode(kernel);
    if (!independent) {
      continue;
    }
    auto iter = nodes_map_.find(kernel.get());
    if (iter != nodes_map_.end()) {
      auto semi_reuse_output_tensors = iter->second->output_tensors_;
      for (auto &tensor : semi_reuse_output_tensors) {
        total_size += tensor->GetAlignedSize();
        tensor->lifelong_value_ = kLifeLongGraphAll;
      }
    }
  }

  MS_LOG(INFO) << "Special Tensor total size: Independent Node output " << total_size;
}

void Somas::SummaryInputProcess(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  bool summary_exist = graph->summary_node_exist();
  if (!summary_exist) {
    return;
  }

  auto summary_nodes = graph->summary_nodes();
  if (summary_nodes.empty()) {
    return;
  }

  size_t total_summary_size = 0;
  for (auto &node_item : summary_nodes) {
    auto node = node_item.second.first;
    size_t index = IntToSize(node_item.second.second);
    auto iter = nodes_map_.find(node.get());
    if (iter != nodes_map_.end()) {
      auto input_node = iter->second;
      if (index < input_node->output_tensors_.size()) {
        auto tensor = iter->second->output_tensors_[index];
        tensor->lifelong_value_ = kLifeLongGraphAll;
        tensor->type_ = kSummaryInput;
        total_summary_size += tensor->GetAlignedSize();
        MS_LOG(INFO) << "Set summary node input tensor's lifelong, node: " << node->fullname_with_scope()
                     << " index: " << index;
      } else {
        MS_LOG(WARNING) << "Index exceed size, node " << node->fullname_with_scope() << " index: " << index
                        << " size: " << input_node->output_tensors_.size();
      }
    } else {
      MS_LOG(WARNING) << "Can't find summary input node " << node->fullname_with_scope() << " index: " << index;
    }
  }

  MS_LOG(INFO) << "Special Tensor total size: SummaryNodes: " << total_summary_size;
}

void Somas::RefNodeProcess(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_cnodes = graph->execution_order();
  size_t total_output_size = 0;
  size_t total_input_size = 0;
  for (const auto &kernel : kernel_cnodes) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    if (kernel_mod == nullptr) {
      MS_LOG(WARNING) << "Kernel mode is NULL Of " << kernel->fullname_with_scope();
      continue;
    }
    auto output_sizes = kernel_mod->GetOutputSizeList();
    size_t output_index = 0;
    for (const auto &size : output_sizes) {
      auto out_index = output_index;
      output_index++;
      session::AnfWithOutIndex out_pair(kernel, out_index);
      if (graph->IsInRefOutputMap(out_pair)) {
        auto origin_pair = graph->GetRefCorrespondOutput(out_pair);
        MS_EXCEPTION_IF_NULL(origin_pair.first);
        auto output_tensor = nodes_map_[kernel.get()]->output_tensors_[out_index];
        MS_EXCEPTION_IF_NULL(output_tensor);
        output_tensor->type_ = kRefNodeOutput;
        total_output_size += size;

        if (AnfAlgo::IsRealCNodeKernel(origin_pair.first)) {
          auto ori_node = origin_pair.first->cast<CNodePtr>();
          auto ori_index = origin_pair.second;
          auto input_tensor = nodes_map_[ori_node.get()]->output_tensors_[ori_index];
          MS_EXCEPTION_IF_NULL(input_tensor);
          input_tensor->type_ = kRefNodeInput;
          total_input_size += input_tensor->aligned_size_;
          std::vector<size_t> refnode_input_output;
          refnode_input_output.push_back(input_tensor->GetId());
          refnode_input_output.push_back(output_tensor->GetId());
          ref_node_constraints_.push_back(refnode_input_output);
          MS_LOG(INFO) << "RefNode: input " << input_tensor->GetId() << " output " << output_tensor->GetId();
        }
      }
    }
  }

  MS_LOG(INFO) << "Special Tensor total size: RefNode: input " << total_input_size << " output " << total_output_size;
}

void Somas::UnReuseNodeProcess(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  vector<string> full_name_list = {};
  if (full_name_list.size() == 0) {
    return;
  }

  auto kernel_cnodes = graph->execution_order();
  for (const auto &kernel : kernel_cnodes) {
    auto full_name = kernel->fullname_with_scope();
    auto iter = std::find(full_name_list.begin(), full_name_list.end(), full_name);
    if (iter != full_name_list.end()) {
      MS_LOG(INFO) << "Set UnReuse Node in somas, Node:" << full_name;
      auto key = kernel.get();
      auto somas_node = nodes_map_[key];
      // input
      auto inputs = somas_node->input_tensors_;
      for (auto &input : inputs) {
        input->lifelong_value_ = kLifeLongGraphAll;
      }

      // output
      auto outputs = somas_node->output_tensors_;
      MS_LOG(INFO) << "Output size of " << kernel->fullname_with_scope() << " is  " << outputs.size();
      for (auto &output : outputs) {
        output->lifelong_value_ = kLifeLongGraphAll;
      }

      // workspace
      auto workspaces = somas_node->workspace_tensors_;
      for (auto &workspace : workspaces) {
        workspace->lifelong_value_ = kLifeLongGraphAll;
      }
    }
  }
}

void Somas::GenContiguousList(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  for (const auto &node : nodes_list_) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->GetType() != kCommunicationNode) {
      continue;
    }

    // Contiguous input
    if ((!node->input_tensors_.empty()) && (!node->input_tensors_[0]->contiguous_)) {
      node->input_tensors_[0]->aligned_size_ += kGapSize;
      node->input_tensors_[node->input_tensors_.size() - 1]->aligned_size_ += kGapSize;
      std::vector<size_t> inputs;
      for (const auto &input_tensor : node->input_tensors_) {
        comm_input_total_size_ += input_tensor->aligned_size_;
        input_tensor->contiguous_ = true;
        inputs.push_back(input_tensor->GetId());
      }
      contiguous_tensors_list_.push_back(inputs);
    }

    // Contiguous output
    if ((!node->output_tensors_.empty()) && (!node->output_tensors_[0]->contiguous_)) {
      node->output_tensors_[0]->aligned_size_ += kGapSize;
      node->output_tensors_[node->output_tensors_.size() - 1]->aligned_size_ += kGapSize;
      std::vector<size_t> outputs;
      for (const auto &output_tensor : node->output_tensors_) {
        comm_output_total_size_ += output_tensor->aligned_size_;
        output_tensor->contiguous_ = true;
        outputs.push_back(output_tensor->GetId());
      }
      contiguous_tensors_list_.push_back(outputs);
    }
  }
}

void Somas::PreprocessingConflicts() {
  // Compute ancestor streams
  for (auto stream : streams_list_) {
    stream->ComputeAncestorStreams();
  }

  // Preset ancestor streams for node
  for (auto node : nodes_list_) {
    node->PresetAncestorStreams(streams_list_);
  }

  // Compute ancestor nodes : needs to be executed in topological order
  for (auto node : nodes_list_) {
    node->ComputeAncestorNodes();
  }

  // Compute MaxDestinationId for between-stream tensors
  for (auto tensor : tensors_list_) {
    if (tensor->IsBetweenStreams()) {
      tensor->ComputeMaxDestinationId();
    }
  }

  // Preprocessing for stream groups
  for (auto group : streams_groups_) {
    vector<SomasStreamPtr> previous_streams;
    for (int64_t stream_id : group) {
      auto it = std::find_if(streams_list_.begin(), streams_list_.end(),
                             [stream_id](const SomasStreamPtr &stream) { return stream->GetId() == stream_id; });
      if (it != streams_list_.end()) {
        for (auto stream : previous_streams) {
          (*it)->ancestor_streams_group_.insert(stream);
        }
        previous_streams.push_back(*it);
      }
    }
  }

  // Atomic: fix any issues on saved lifetimes of tensors
  for (auto tensor : tensors_list_) {
    MS_EXCEPTION_IF_NULL(tensor);
    for (auto node : tensor->destinations_) {
      MS_EXCEPTION_IF_NULL(node);
      MS_EXCEPTION_IF_NULL(tensor->GetSourceNode());
      if (tensor->GetSourceNode()->GetId() > node->GetId()) {
        tensor->lifetime_.start_ = node->GetId();
      }
    }
    MS_EXCEPTION_IF_NULL(tensor->GetSourceNode());
    if (tensor->GetSourceNode()->GetId() > tensor->lifetime_.end_) {
      tensor->lifetime_.end_ = tensor->GetSourceNode()->GetId();
    }
  }
}

void Somas::ComputeConflictPairs() {
  if (tensors_list_.empty()) {
    MS_LOG(INFO) << "No Tensor for Conflict computing";
    return;
  }

  MS_LOG(INFO) << "Start Preprocessing Conflicts";
  PreprocessingConflicts();
  MS_LOG(INFO) << "End Preprocessing Conflicts";

  MS_LOG(INFO) << "Start Conflict Computing (Bitset Model)";
  auto start_conflict = std::chrono::system_clock::now();
  std::sort(nodes_list_.begin(), nodes_list_.end(), NodeSort);

  // Loop to add edges within each stream (node order within stream)
  for (const auto &stream : streams_list_) {
    auto &nodes = stream->nodes_;
    std::sort(nodes.begin(), nodes.end(), NodeSort);
    for (size_t i = 1; i < nodes.size(); i++) {
      const auto &previous_node = nodes[i - 1];
      const auto &current_node = nodes[i];
      current_node->ancestor_nodes_.insert(previous_node);
    }
  }

  // Loop to add edges from end to beginning of next group
  for (const auto &group : streams_groups_) {
    for (size_t i = 1; i < group.size(); i++) {
      int64_t previous_stream = group[i - 1];
      int64_t current_stream = group[i];

      auto it =
        std::find_if(streams_list_.begin(), streams_list_.end(),
                     [previous_stream](const SomasStreamPtr &stream) { return stream->GetId() == previous_stream; });
      if (it == streams_list_.end()) {
        continue;
      }
      auto &last_node_in_prev_stream = (*it)->nodes_.back();

      it = std::find_if(streams_list_.begin(), streams_list_.end(),
                        [current_stream](const SomasStreamPtr &stream) { return stream->GetId() == current_stream; });
      if (it == streams_list_.end()) {
        continue;
      }
      auto &first_node_in_cur_stream = (*it)->nodes_.front();

      first_node_in_cur_stream->ancestor_nodes_.insert(last_node_in_prev_stream);
    }
  }

  // Loop to avoid tensors with empty destinations (add itself)
  for (const auto &tensor : tensors_list_) {
    if (tensor->destinations_.size() == 0) {
      tensor->destinations_.insert(tensor->GetSourceNode());
    }
  }

  MS_LOG(INFO) << "Start Bitset";
  std::vector<DynamicBitSet> nodes_dependency;

  size_t count = nodes_list_.back()->GetId() + 1;
  for (size_t i = 0; i < count; i++) {
    nodes_dependency.emplace_back(count);
  }

  MS_LOG(INFO) << "Start Path Computing";
  // Loop to compute ancestor paths via bitset for time dependence
  for (const auto &node : nodes_list_) {
    for (const auto &ancestor : node->ancestor_nodes_) {
      nodes_dependency[node->GetId()].SetBitTrue(ancestor->GetId());
      Union(&nodes_dependency[node->GetId()], &nodes_dependency[ancestor->GetId()]);
    }
  }
  MS_LOG(INFO) << "End Path Computing";

  MS_LOG(INFO) << "Start Tensor Relation Computing";
  count = tensors_list_.back()->GetId() + 1;
  for (size_t i = 0; i < count; i++) {
    reuse_matrix_.emplace_back(count);
  }

  if (tensors_list_.size() < kParallelComputeSizeThreshold) {
    ComputeMultiTensorConflicts(tensors_list_, tensors_list_, nodes_dependency, &reuse_matrix_);
  } else {
    MS_LOG(INFO) << "Tensor Num " << tensors_list_.size() << " is larger than " << kParallelComputeSizeThreshold;
    MS_LOG(INFO) << "Enter Multi-Thread Mode...";
    size_t process_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
    MS_LOG(INFO) << "Threads Num is " << process_num;

    size_t start_index = 0;
    size_t total_size = tensors_list_.size();
    size_t job_size = total_size / process_num;
    if (job_size == 0) {
      job_size = total_size;
    }
    std::vector<common::Task> tasks;
    while (start_index < total_size) {
      size_t end_index = (start_index + job_size) > total_size ? total_size : start_index + job_size;
      auto jobs = std::vector<SomasTensorPtr>(tensors_list_.begin() + start_index, tensors_list_.begin() + end_index);
      auto task = [this, jobs, &nodes_dependency]() {
        this->ComputeMultiTensorConflicts(jobs, tensors_list_, nodes_dependency, &reuse_matrix_);
        return common::SUCCESS;
      };
      tasks.emplace_back(task);
      start_index += job_size;
    }

    common::ThreadPool::GetInstance().SyncRun(tasks);
  }
  MS_LOG(INFO) << "End Tensor Relation Computing";
  auto end_conflict = std::chrono::system_clock::now();
  MS_LOG(INFO) << "End Conflict Computing (Bitset Model)(time taken "
               << std::chrono::duration_cast<std::chrono::milliseconds>(end_conflict - start_conflict).count() << "ms)";
}

void Somas::ComputeMultiTensorConflicts(const std::vector<SomasTensorPtr> &calc_tensors_list,
                                        const std::vector<SomasTensorPtr> &all_tensors_list,
                                        const vector<DynamicBitSet> &nodes_dependency,
                                        std::vector<DynamicBitSet> *tensor_relation) const {
  auto start = std::chrono::system_clock::now();
  MS_LOG(INFO) << "Start Computing Conflicts Pairs, tensors list size is " << calc_tensors_list.size();
  for (size_t i = 0; i < calc_tensors_list.size(); i++) {
    auto calc_tensor = calc_tensors_list[i];
    if (calc_tensor->IsLifelong() || calc_tensor->IsRefOverlap() || calc_tensor->GetAlignedSize() == 0) {
      continue;
    }

    ComputeOneTensorConflicts(calc_tensor, all_tensors_list, nodes_dependency, tensor_relation);
  }
  auto end = std::chrono::system_clock::now();
  MS_LOG(INFO) << "End Computing Conflicts Pairs (time taken "
               << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms)";
}

void Somas::ComputeOneTensorConflicts(const std::shared_ptr<SomasTensor> &calc_tensor,
                                      const std::vector<SomasTensorPtr> &all_tensors_list,
                                      const vector<DynamicBitSet> &nodes_dependency,
                                      std::vector<DynamicBitSet> *tensor_relation) const {
  for (size_t j = 0; j < all_tensors_list.size(); j++) {
    auto target_tensor = all_tensors_list[j];
    if (calc_tensor == target_tensor || target_tensor->IsLifelong() || target_tensor->IsRefOverlap() ||
        target_tensor->GetAlignedSize() == 0) {
      continue;
    }
    size_t calc_src_node = calc_tensor->GetSourceNode()->GetId();
    size_t target_src_node = target_tensor->GetSourceNode()->GetId();
    if (calc_src_node == target_src_node) {
      continue;
    }
    if ((*tensor_relation)[calc_tensor->GetId()].IsBitTrue(target_tensor->GetId()) ||
        (*tensor_relation)[target_tensor->GetId()].IsBitTrue(calc_tensor->GetId())) {
      continue;
    }

    bool reuse = true;
    // check calc_tensor's all consumers is target_tensor's source node's dependency or not
    for (const auto &dst_node : calc_tensor->destinations_) {
      if (nodes_dependency[target_src_node].IsBitTrue(dst_node->GetId()) == false) {
        // calc_tensor's consumer is not in target_tensor's source node's dependency, not sure this consumer is done or
        // not when target_tensor produced
        reuse = false;
        break;
      } else if (target_src_node == dst_node->GetId()) {
        // calc_tensor is target_tensor's source node's input, can't reuse
        reuse = false;
        break;
      } else {
        // calc_tensor's consumer is in target_tensor's source node's dependency, this consumer is done when
        // target_tensor produced
        reuse = true;
      }
    }

    if (reuse) {
      // calc_tensor and target_tensor have dependencies so they can reuse each other
      (*tensor_relation)[calc_tensor->GetId()].SetBitTrue(target_tensor->GetId());
      (*tensor_relation)[target_tensor->GetId()].SetBitTrue(calc_tensor->GetId());
    }
  }
}

bool Somas::NodeSort(SomasNodePtr node1, SomasNodePtr node2) { return node1->GetId() < node2->GetId(); }

bool Somas::Assign(const session::KernelGraph *graph) {
  if (tensors_list_.empty()) {
    MS_LOG(INFO) << "No Tensor for Assigner";
    return true;
  }

  // Ref Node Preprocessing
  MS_LOG(INFO) << "Start Solving Preprocessing for Ref Node";
  std::map<size_t, size_t> contiguous_ref_map;
  for (auto ref_node_list : ref_node_constraints_) {
    // Count contiguous tensors in ref list
    size_t contiguous_in_ref_list = std::count_if(ref_node_list.begin(), ref_node_list.end(),
                                                  [this](size_t tid) { return tensors_map_[tid]->contiguous_; });
    // Keep all constraints for first tensor in list
    size_t tid_0 = ref_node_list[0];
    for (SomasTensorPtr tensor : tensors_list_) {
      if (reuse_matrix_[tid_0].IsBitTrue(tensor->GetId()) == false) {
        continue;
      }
      for (size_t tid : ref_node_list) {
        if (reuse_matrix_[tid].IsBitTrue(tensor->GetId()) == false) {
          reuse_matrix_[tid_0].SetBitFalse(tensor->GetId());
          reuse_matrix_[tensor->GetId()].SetBitFalse(tid_0);
          break;
        }
      }
    }
    // Set rest to size 0, so that solver ignores them (if not contiguous)
    for (size_t i = 1; i < ref_node_list.size(); ++i) {
      if (!tensors_map_[ref_node_list[i]]->contiguous_) {
        tensors_map_[ref_node_list[i]]->aligned_size_ = 0;
      }
    }
    // Keep info about contiguous and check for errors
    if (ref_node_list.size() > 2 && contiguous_in_ref_list > 0) {
      MS_LOG(WARNING) << "Ref node of size greater than two with at least one contiguous tensor in";
    }
    if (ref_node_list.size() == 2 && contiguous_in_ref_list == 1) {
      MS_LOG(WARNING) << "Ref node of size two with only one contiguous tensor" << ref_node_list[0] << ":"
                      << tensors_map_[ref_node_list[0]]->contiguous_ << ", " << ref_node_list[1] << ":"
                      << tensors_map_[ref_node_list[1]]->contiguous_;
    }
    if (ref_node_list.size() == 2 && contiguous_in_ref_list == 2) {
      contiguous_ref_map[ref_node_list[0]] = ref_node_list[1];
    }
  }
  // Handle contiguous ref node (remove ref from contiguous_tensors_list_)
  std::map<size_t, size_t> contiguous_ref_list_map;
  std::map<size_t, std::map<size_t, std::set<size_t>>> contiguous_ref_list_error_check_map;
  for (auto ref_pair : contiguous_ref_map) {
    size_t ref_first = ref_pair.first;
    size_t ref_second = ref_pair.second;
    bool found_first = false;
    bool found_second = false;
    size_t index_first = 0;
    size_t index_second = 0;
    size_t index_in_list_first = 0;
    size_t index_in_list_second = 0;
    for (size_t index = 0; index < contiguous_tensors_list_.size() && (!found_first || !found_second); index++) {
      if (!found_first) {
        auto iterator_first =
          std::find(contiguous_tensors_list_[index].begin(), contiguous_tensors_list_[index].end(), ref_first);
        if (iterator_first != contiguous_tensors_list_[index].end()) {
          index_first = index;
          index_in_list_first = iterator_first - contiguous_tensors_list_[index].begin();
          found_first = true;
        }
      }
      if (!found_second) {
        auto iterator_second =
          std::find(contiguous_tensors_list_[index].begin(), contiguous_tensors_list_[index].end(), ref_second);
        if (iterator_second != contiguous_tensors_list_[index].end()) {
          index_second = index;
          index_in_list_second = iterator_second - contiguous_tensors_list_[index].begin();
          found_second = true;
        }
      }
    }
    if (!found_first) {
      MS_LOG(WARNING) << "Contiguous ref tensor " << ref_first << " not found in any contiguous list";
    }
    if (!found_second) {
      MS_LOG(WARNING) << "Contiguous ref tensor " << ref_second << " not found in any contiguous list";
    }
    if (contiguous_ref_list_map.find(index_first) == contiguous_ref_list_map.end() ||
        contiguous_ref_list_map[index_first] == index_second) {
      contiguous_ref_list_map[index_first] = index_second;
      // Checking for error cases
      if (index_in_list_first != index_in_list_second) {
        MS_LOG(WARNING) << "Inconsistency in contiguous ref: tensor " << ref_first << " in position "
                        << index_in_list_first << " of contiguous list " << index_first << " and tensor " << ref_second
                        << " in position " << index_in_list_second << " of contiguous list " << index_second;
      }
      contiguous_ref_list_error_check_map[index_first][index_second].insert(index_in_list_first);
    } else {  // contiguous_ref_list_map.find(index_first) != contiguous_ref_list_map.end() &&
      // contiguous_ref_list_map[index_first] != index_second
      MS_LOG(WARNING) << "Contiguous list " << index_first << " associated (ref node) with two other contiguous lists: "
                      << contiguous_ref_list_map[index_first] << " and " << index_second;
    }
  }

  for (auto check_list_pair : contiguous_ref_list_error_check_map) {
    auto first_list = check_list_pair.first;
    auto index_set_map = check_list_pair.second;
    for (auto index_set : index_set_map) {
      auto second_list = index_set.first;
      if (contiguous_tensors_list_[first_list].size() != contiguous_tensors_list_[second_list].size()) {
        MS_LOG(WARNING) << "Contiguous lists " << first_list << " and " << second_list
                        << " considered in ref do not have the same size";
      }
      for (size_t x = 0; x < contiguous_tensors_list_[second_list].size(); x++) {
        if (contiguous_ref_list_error_check_map[first_list][second_list].count(x) == 0) {
          MS_LOG(WARNING) << "Contiguous lists " << first_list << " and " << second_list
                          << " considered in ref: ref pair at in-lists index " << x << " has not been considered";
        }
      }
    }
  }

  std::set<vector<size_t>> contiguous_tensors_list_to_remove;
  for (auto ref_list_pair : contiguous_ref_list_map) {
    contiguous_tensors_list_to_remove.insert(contiguous_tensors_list_[ref_list_pair.second]);
  }
  vector<vector<size_t>> contiguous_tensors_list_removed_ref = contiguous_tensors_list_;
  for (auto contiguous_list : contiguous_tensors_list_to_remove) {
    auto iterator = std::find(contiguous_tensors_list_removed_ref.begin(), contiguous_tensors_list_removed_ref.end(),
                              contiguous_list);
    if (iterator != contiguous_tensors_list_removed_ref.end()) {
      contiguous_tensors_list_removed_ref.erase(iterator);
    } else {
      MS_LOG(WARNING) << "Could not find contiguous list to remove for ref";
    }
  }
  MS_LOG(INFO) << "End Solving Preprocessing for Ref Node";

  // Ref Overlap Preprocessing
  MS_LOG(INFO) << "Start Solving Preprocessing for Ref Overlap";
  // In ConflictComputing(), by use of ref_overlap_ flag, each tensor in a ref_overlap_list has all entries 1 in
  // cannot_reuse_ array Here, we allow reuse only among tensors in same list
  for (auto ref_overlap_list : ref_overlap_constraints_) {
    for (size_t tid_1 : ref_overlap_list) {
      for (size_t tid_2 : ref_overlap_list) {
        reuse_matrix_[tid_1].SetBitTrue(tid_2);
        reuse_matrix_[tid_2].SetBitTrue(tid_1);
      }
    }
  }
  MS_LOG(INFO) << "End Solving Preprocessing for Ref Overlap";

#ifdef SOMAS_DEBUG
  // Compute number of constraints for each tensor
  auto tensors_num = tensors_list_.size();
  for (auto tensor1 : tensors_list_) {
    auto ones_num = reuse_matrix_[tensor1->GetId()].CountOnesNum();
    tensor1->num_constraints_ = tensors_num - ones_num;
  }
#endif

  // Prepare solver info
  MS_LOG(INFO) << "Start Loop to create solver info";
  for (auto tensor : tensors_list_) {
    if (tensor->GetSolverTensorDesc() != nullptr) {
      SomasSolverTensorDescPtr pSolverTensor = tensor->GetSolverTensorDesc();
      solver_tensor_desc_list_.insert(
        std::pair<size_t, SomasSolverTensorDescPtr>(pSolverTensor->index_, pSolverTensor));
    }
  }
  MS_LOG(INFO) << "End Loop to create solver info";

  MS_LOG(INFO) << "Start Solving";
  if (solver_tensor_desc_list_.empty()) {
    MS_LOG(INFO) << "solver_tensor_desc_list is empty.";
    return true;
  }

  somas_solver_ = std::make_shared<SomasSolverPre>();
  auto status = somas_solver_->Solving(graph, &solver_tensor_desc_list_, &reuse_matrix_,
                                       contiguous_tensors_list_removed_ref, false);
  MS_LOG(INFO) << "End Solving";
  if (status != SUCCESS) {
    GenGraphStatisticInfo();
    MS_LOG(EXCEPTION) << "SOMAS Solving Failed.";
  }

  // Update solver_tensor_desc offset to tensors list
  for (const auto &tensor : tensors_list_) {
    tensor->SetOffset();
  }

  // Ref Node Postprocessing
  MS_LOG(INFO) << "\nStart Solving Postprocessing for Ref Node";
  // Set offset for rest of ref node list (ignored by solver due to ref node preprocessing)
  for (auto ref_node_list : ref_node_constraints_) {
    for (size_t i = 1; i < ref_node_list.size(); ++i) {
      tensors_map_[ref_node_list[i]]->offset_ = tensors_map_[ref_node_list[0]]->offset_;
    }
  }
  // Handle contiguous ref node
  for (auto ref_list_pair : contiguous_ref_list_map) {
    size_t index_first = ref_list_pair.first;
    size_t index_second = ref_list_pair.second;
    for (size_t x = 0; x < contiguous_tensors_list_[index_second].size(); x++) {
      tensors_map_[contiguous_tensors_list_[index_second][x]]->offset_ =
        tensors_map_[contiguous_tensors_list_[index_first][x]]->offset_;
    }
  }
  MS_LOG(INFO) << "\nEnd Solving Postprocessing for Ref Node";

  // Contiguous gaps postprocessing
  for (auto list : contiguous_tensors_list_) {
    tensors_map_[list[0]]->offset_ += kGapSize;
  }

  // Set mem_offset_ value by solver result
  mem_offset_ = static_cast<size_t>(somas_solver_->GetMaxOffset());

  return true;
}

std::string Somas::GetSplitName(const std::string &scope_name) const {
  auto indx = scope_name.rfind('/');
  if (indx == std::string::npos) {
    return scope_name;
  } else {
    if (indx < scope_name.size() - 1) {
      auto split_name = scope_name.substr(indx + 1);
      return split_name;
    }
    return scope_name;
  }
}

void Somas::DumpSomasInfoIR(const string filename) {
  if (filename.size() > PATH_MAX) {
    MS_LOG(ERROR) << "File path " << filename << " is too long.";
    return;
  }
  auto real_path = Common::GetRealPath(filename);
  if (!real_path.has_value()) {
    MS_LOG(ERROR) << "Get real path failed. path=" << filename;
    return;
  }

  ChangeFileMode(real_path.value(), S_IRWXU);
  std::ofstream ofs(real_path.value());

  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open dump file '" << real_path.value() << "' failed!";
    return;
  }

  ofs << "All Parameters:\n\n";
  ofs << "index:"
      << "\tsize:"
      << "\tstart_addr:"
      << "\tsource node name:"
      << "\tnode out index:\n";

  for (const auto &param : parameters_list_) {
    ofs << "%" << param->id_ << "P"
        << "\t"
        << "#" << param->size_ << "S"
        << "\t"
        << "&" << param->addr_ << "\t" << param->source_node_->fullname_with_scope() << "\t" << param->output_index_
        << "\n";
  }

  ofs << "\n\nAll Tensors:\n\n";
  ofs << "index:"
      << "\tsize:"
      << "\treal_size:"
      << "\toffset:"
      << "\taddr:"
      << "\ttype:"
      << "\tlifelong:"
      << "\tlife_start:"
      << "\tlife_end:"
      << "\tsource node name:\n";

  for (const auto &tensor : tensors_list_) {
    auto scope_name = tensor->GetSourceNode()->scope_full_name_;
    std::string split_name = GetSplitName(scope_name);
    ofs << "%" << tensor->GetId() << "T"
        << "\t"
        << "#" << tensor->GetAlignedSize() << "S"
        << "\t"
        << "#" << tensor->GetOriginalSize() << "S"
        << "\t"
        << "&" << tensor->GetOffset() << ""
        << "\t"
        << "&" << static_cast<void *>(tensor->GetOffset() + mem_base_addr_) << "\t"
        << tensor_type_name_map[tensor->type_] << "\t" << tensor->IsLifelong() << "\t" << tensor->lifetime_.start_
        << "\t" << tensor->lifetime_.end_ << "\t" << split_name << "\n";
  }

  ofs << "\n\nAll Nodes:\n\n";
  for (const auto &node : nodes_list_) {
    auto scope_name = node->scope_full_name_;
    std::string split_name = GetSplitName(scope_name);
    ofs << "$" << node->GetId() << "\t" << split_name << "\t" << static_cast<int>(node->GetType()) << "\t";
    vector<std::pair<string, size_t>> input_list;
    std::transform(
      node->input_tensors_.begin(), node->input_tensors_.end(), std::back_inserter(input_list),
      [](SomasTensorPtr in) -> std::pair<string, size_t> { return std::make_pair("Tensor", in->GetId()); });
    for (const auto &param : node->input_parameters_map_) {
      auto index = param.first;
      auto iter = input_list.begin();
      std::advance(iter, index);
      input_list.insert(iter, std::make_pair("Parameter", param.second->id_));
    }
    ofs << "inputs[";
    for (const auto &in : input_list) {
      if (in.first == "Tensor") {
        ofs << "%" << in.second << "T"
            << ", ";
      } else if (in.first == "Parameter") {
        ofs << "%" << in.second << "P"
            << ", ";
      }
    }
    ofs << "]";
    ofs << "\toutputs[";
    for (const auto &out : node->output_tensors_) {
      ofs << "%" << out->GetId() << "T"
          << ", ";
    }
    ofs << "]";
    ofs << "\tworkspace[";
    for (const auto &wk : node->workspace_tensors_) {
      ofs << "%" << wk->GetId() << "T"
          << ", ";
    }
    ofs << "]";
    ofs << "\tstreamID["
        << "@" << node->GetStream()->GetId() << "]\n";
  }

  ofs << "\n\nAll Stream Groups:\n\n";
  for (const auto &stream_group : streams_groups_) {
    for (const auto &stream : stream_group) {
      ofs << "stm" << stream << " ";
    }
    ofs << "\n";
  }

  if (!ref_node_constraints_.empty()) {
    ofs << "\n\nAll Ref Node Info:\n\n";
    for (const auto &ref_in_out : ref_node_constraints_) {
      ofs << "refnode input-output:";
      for (const auto &item : ref_in_out) {
        ofs << "%" << item << "T ";
      }
      ofs << "\n";
    }
  }
}

void Somas::DumpOfflineIR(const string filename) {
  MS_LOG(INFO) << "Printing somas-log-from-graph log: " << filename;
  if (filename.size() > PATH_MAX) {
    MS_LOG(ERROR) << "File path " << filename << " is too long.";
    return;
  }

  auto real_path = Common::GetRealPath(filename);
  if (!real_path.has_value()) {
    MS_LOG(ERROR) << "Get real path failed. path=" << filename;
    return;
  }

  ChangeFileMode(real_path.value(), S_IRWXU);
  std::ofstream ofs(real_path.value());

  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open dump file '" << real_path.value() << "' failed!";
    return;
  }

  for (auto tensor : tensors_list_) {
    if (tensor->IsOutputOnly() || tensor->type_ == TensorType::kRefNodeOutput) {
      ofs << "Somas EDGE ERROR src=n" << tensor->GetSourceNode()->GetId()
          << ", srcstm=" << tensor->GetSourceStream()->GetId() << ", dst=nc"
          << ", dststm=nc"
          << ", workspace=0, size=" << tensor->GetOriginalSize()
          << ", lifelong=" << static_cast<int>(tensor->lifelong_value_) << ", tid=" << tensor->GetId()
          << ", start=" << tensor->lifetime_.start_ << ", end=" << tensor->lifetime_.end_ << std::endl;
    } else {
      std::map<size_t, size_t> dest_infos;
      for (SomasNodePtr dest_node : tensor->destinations_) {
        dest_infos.insert(std::make_pair(dest_node->GetId(), dest_node->GetStream()->GetId()));
      }

      for (auto dest_info : dest_infos) {
        ofs << "Somas EDGE src=n" << tensor->GetSourceNode()->GetId()
            << ", srcstm=" << tensor->GetSourceStream()->GetId() << ", dst=n" << dest_info.first
            << ", dststm=" << dest_info.second << ", workspace=" << static_cast<int>(tensor->type_ == kWorkspace)
            << ", size=" << tensor->GetOriginalSize() << ", lifelong=" << static_cast<int>(tensor->lifelong_value_)
            << ", tid=" << tensor->GetId() << ", start=" << tensor->lifetime_.start_
            << ", end=" << tensor->lifetime_.end_ << std::endl;
      }
    }
  }
  for (vector<size_t> tList : contiguous_tensors_list_) {
    ofs << "Somas CONTIGUOUS";
    for (size_t tid : tList) {
      ofs << " " << tid;
    }
    ofs << std::endl;
  }
  for (const auto &group : streams_groups_) {
    ofs << "Somas GROUP";
    for (int64_t sid : group) {
      ofs << " " << sid;
    }
    ofs << std::endl;
  }
  ofs.close();
}

void Somas::DumpSomasMemoryIR(const string filename) {
  if (filename.size() > PATH_MAX) {
    MS_LOG(ERROR) << "File path " << filename << " is too long.";
    return;
  }

  auto real_path = Common::GetRealPath(filename);
  if (!real_path.has_value()) {
    MS_LOG(ERROR) << "Get real path failed. path=" << filename;
    return;
  }

  ChangeFileMode(real_path.value(), S_IRWXU);
  std::ofstream ofs(real_path.value());

  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open dump file '" << real_path.value() << "' failed!";
    return;
  }

  std::map<size_t, size_t> mem_map;
  for (auto tensor : tensors_list_) {
    mem_map[tensor->GetOffset()] = 0;
  }

  size_t num = 0;
  for (auto iter = mem_map.begin(); iter != mem_map.end(); ++iter, ++num) {
    iter->second = num;
  }

  std::map<size_t, std::map<size_t, SomasTensorPtr>> mem_list;

  for (const auto &tensor : tensors_list_) {
    size_t key = tensor->offset_;
    auto iter = mem_list.find(key);
    if (iter == mem_list.end()) {
      std::map<size_t, SomasTensorPtr> id_tensor_map;
      id_tensor_map[tensor->GetId()] = tensor;
      mem_list[key] = id_tensor_map;
    } else {
      iter->second[tensor->GetId()] = tensor;
    }
  }

  ofs << "mem_id:"
      << "\tstart_offset:"
      << "\tend_offset:"
      << "\ttensor_id:"
      << "\torigin_size:"
      << "\talign_size:"
      << "\tstart_addr:"
      << "\tend_addr:"
      << "\ttype:"
      << "\tsrc_node:"
      << "\tsrc_stm_id:"
      << "lifetime_start\t"
      << "lifetime_end\n";

  for (const auto &mem : mem_list) {
    auto id_tensor_map = mem.second;
    for (const auto &id_tensor : id_tensor_map) {
      auto tensor = id_tensor.second;
      std::string scope_name;
      size_t src_stm_id = 0xffff;
      if (tensor->GetSourceNode() != nullptr) {
        scope_name = tensor->GetSourceNode()->scope_full_name_;
        src_stm_id = tensor->GetSourceNode()->GetStream()->GetId();
      } else {
        scope_name = "Somas Tensor";
      }

      std::string split_name = GetSplitName(scope_name);
      ofs << "#" << mem_map[tensor->GetOffset()] << "\t" << tensor->GetOffset() << "\t"
          << tensor->GetOffset() + tensor->GetAlignedSize() << "\t%" << tensor->GetId() << "T\t"
          << tensor->GetOriginalSize() << "\t" << tensor->GetAlignedSize() << "\t&"
          << static_cast<void *>(tensor->GetOffset() + mem_base_addr_) << "\t&"
          << static_cast<void *>(tensor->GetOffset() + mem_base_addr_ + tensor->GetAlignedSize()) << "\t"
          << tensor_type_name_map[tensor->type_] << "\t" << split_name << "\tstm" << src_stm_id << "\t"
          << tensor->lifetime_.start_ << "\t" << tensor->lifetime_.end_ << "\n";
    }
  }
}

size_t Somas::CalcLowerBound() const {
  size_t max_node_id = std::accumulate(tensors_list_.begin(), tensors_list_.end(), 0, [](size_t max_id, auto tensor) {
    return std::max(max_id, tensor->lifetime_.end_);
  });

  std::map<size_t, size_t> lifetime_lb;
  for (size_t time = 0; time <= max_node_id; time++) {
    lifetime_lb[time] = 0;
  }

  size_t lower, upper;
  for (auto tensor : tensors_list_) {
    if (tensor->lifelong_value_ == kLifeLongGraphAll) {
      lower = 0;
      upper = max_node_id;
    } else {
      lower = tensor->lifetime_.start_;
      upper = tensor->lifetime_.end_;
    }

    for (size_t time = lower; time <= upper; time++) {
      lifetime_lb[time] += tensor->GetAlignedSize();
    }
  }

  size_t max_lifetime = 0;
  for (size_t time = 0; time <= max_node_id; time++) {
    if (max_lifetime < lifetime_lb[time]) {
      max_lifetime = lifetime_lb[time];
    }
  }
  return max_lifetime;
}

void Somas::GenGraphStatisticInfo() {
  lower_bound_ = CalcLowerBound();
  for (const auto &tensor : tensors_list_) {
    upper_bound_ += tensor->aligned_size_;
    if (tensor->type_ == kWorkspace) {
      workspace_total_size_ += tensor->aligned_size_;
    }
    if (tensor->lifelong_value_ == kLifeLongGraphAll) {
      lifelong_all_total_size_ += tensor->aligned_size_;
    } else if (tensor->lifelong_value_ == kLifeLongGraphStart) {
      lifelong_start_total_size_ += tensor->aligned_size_;
    } else if (tensor->lifelong_value_ == kLifeLongGraphEnd) {
      lifelong_end_total_size_ += tensor->aligned_size_;
    }
  }

  const double giga = 1024. * 1024. * 1024.;
  MS_LOG(INFO) << "Lower Bound: " << lower_bound_ << " (" << lower_bound_ / giga
               << " GB), Upper Bound: " << upper_bound_ << " (" << upper_bound_ / giga << " GB)";

  MS_LOG(INFO) << "\nTotal Dynamic Size (Upper Bound):\t" << upper_bound_ << "\n"
               << "Theoretical Optimal Size (Lower Bound):\t" << lower_bound_ << "\n"
               << "Total Workspace Size:\t" << workspace_total_size_ << "\n"
               << "Total Communication Input Tensor Size:\t" << comm_input_total_size_ << "\n"
               << "Total Communication Output Tensor Size:\t" << comm_output_total_size_ << "\n"
               << "Total LifeLong All Tensor Size:\t" << lifelong_all_total_size_ << "\n"
               << "Total LifeLong Start Tensor Size:\t" << lifelong_start_total_size_ << "\n"
               << "Total LifeLong End Tensor Size:\t" << lifelong_end_total_size_ << "\n"
               << "Reused Size(Allocate Size):\t" << GetTotalMemSize() << "\n\n\n";
}

uint8_t *Somas::GetNodeOutputPtr(const AnfNodePtr &node, size_t index) const {
  auto key = node.get();
  auto iter = nodes_map_.find(key);
  uint8_t *ptr = nullptr;
  if (iter != nodes_map_.end()) {
    if (index >= iter->second->output_tensors_.size()) {
      MS_LOG(EXCEPTION) << "index:[" << index << "] is larger than it's workspace size:["
                        << iter->second->output_tensors_.size() << "]";
    }
    auto output_tensor = iter->second->output_tensors_[index];
    ptr = mem_base_addr_ + output_tensor->offset_;
  } else {
    MS_LOG(EXCEPTION) << "node [" << AnfAlgo::GetCNodeName(node) << "] don't exist in nodes_map";
  }
  return ptr;
}

uint8_t *Somas::GetNodeWorkSpacePtr(const AnfNodePtr &node, size_t index) const {
  auto key = node.get();
  auto iter = nodes_map_.find(key);
  uint8_t *ptr = nullptr;
  if (iter != nodes_map_.end()) {
    if (index >= iter->second->workspace_tensors_.size()) {
      MS_LOG(EXCEPTION) << "index:[" << index << "] is larger than it's workspace size:["
                        << iter->second->workspace_tensors_.size() << "]";
    }
    auto workspace_tensor = iter->second->workspace_tensors_[index];
    ptr = mem_base_addr_ + workspace_tensor->offset_;
  }
  return ptr;
}
}  // namespace somas
}  // namespace mindspore
