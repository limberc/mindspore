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

#include "minddata/dataset/engine/ir/datasetops/source/cifar10_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/cifar_op.h"

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

// Constructor for Cifar10Node
Cifar10Node::Cifar10Node(const std::string &dataset_dir, const std::string &usage, std::shared_ptr<SamplerObj> sampler,
                         std::shared_ptr<DatasetCache> cache)
    : MappableSourceNode(std::move(cache)), dataset_dir_(dataset_dir), usage_(usage), sampler_(sampler) {}

std::shared_ptr<DatasetNode> Cifar10Node::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->Copy();
  auto node = std::make_shared<Cifar10Node>(dataset_dir_, usage_, sampler, cache_);
  return node;
}

void Cifar10Node::Print(std::ostream &out) const {
  out << Name() + "(cache:" + ((cache_ != nullptr) ? "true" : "false") + ")";
}

Status Cifar10Node::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("Cifar10Node", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("Cifar10Node", sampler_));

  RETURN_IF_NOT_OK(ValidateStringValue("Cifar10Node", usage_, {"train", "test", "all"}));

  return Status::OK();
}

// Function to build CifarOp for Cifar10
Status Cifar10Node::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));

  RETURN_IF_NOT_OK(AddCacheOp(node_ops));

  node_ops->push_back(std::make_shared<CifarOp>(CifarOp::CifarType::kCifar10, usage_, num_workers_, rows_per_buffer_,
                                                dataset_dir_, connector_que_size_, std::move(schema),
                                                std::move(sampler_->Build())));

  return Status::OK();
}

// Get the shard id of node
Status Cifar10Node::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

// Get Dataset size
Status Cifar10Node::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                   int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows, sample_size;
  RETURN_IF_NOT_OK(CifarOp::CountTotalRows(dataset_dir_, usage_, true, &num_rows));
  sample_size = sampler_->Build()->CalculateNumSamples(num_rows);
  *dataset_size = sample_size;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
