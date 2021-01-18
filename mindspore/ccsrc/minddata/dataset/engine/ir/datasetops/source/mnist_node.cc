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

#include "minddata/dataset/engine/ir/datasetops/source/mnist_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/mnist_op.h"

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

MnistNode::MnistNode(std::string dataset_dir, std::string usage, std::shared_ptr<SamplerObj> sampler,
                     std::shared_ptr<DatasetCache> cache)
    : MappableSourceNode(std::move(cache)), dataset_dir_(dataset_dir), usage_(usage), sampler_(sampler) {}

std::shared_ptr<DatasetNode> MnistNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->Copy();
  auto node = std::make_shared<MnistNode>(dataset_dir_, usage_, sampler, cache_);
  return node;
}

void MnistNode::Print(std::ostream &out) const { out << Name(); }

Status MnistNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("MnistNode", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("MnistNode", sampler_));

  RETURN_IF_NOT_OK(ValidateStringValue("MnistNode", usage_, {"train", "test", "all"}));

  return Status::OK();
}

Status MnistNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));
  RETURN_IF_NOT_OK(AddCacheOp(node_ops));

  node_ops->push_back(std::make_shared<MnistOp>(usage_, num_workers_, rows_per_buffer_, dataset_dir_,
                                                connector_que_size_, std::move(schema), std::move(sampler_->Build())));

  return Status::OK();
}

// Get the shard id of node
Status MnistNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

// Get Dataset size
Status MnistNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                 int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows, sample_size;
  RETURN_IF_NOT_OK(MnistOp::CountTotalRows(dataset_dir_, usage_, &num_rows));
  sample_size = sampler_->Build()->CalculateNumSamples(num_rows);
  *dataset_size = sample_size;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

Status MnistNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["dataset_dir"] = dataset_dir_;
  args["usage"] = usage_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  *out_json = args;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
