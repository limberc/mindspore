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
#include "tools/converter/parser/tf/tf_batchnorm_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"

namespace mindspore {
namespace lite {
STATUS TFBatchNormParser::Parse(const tensorflow::NodeDef &tf_op,
                                const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                PrimitiveC **primitiveC, std::vector<std::string> *inputs, int *output_size) {
  MS_LOG(INFO) << "TF BatchNormParser";
  if (primitiveC == nullptr || output_size == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_NULL_PTR;
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return RET_NULL_PTR;
  }
  auto attr = std::make_unique<schema::FusedBatchNormT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }
  tensorflow::AttrValue attr_value;
  TensorFlowUtils::FindAttrValue(tf_op, "epsilon", &attr_value);
  attr->epsilon = attr_value.f();

  primitive->value.type = schema::PrimitiveType_FusedBatchNorm;
  primitive->value.value = attr.release();
  *primitiveC = PrimitiveC::Create(primitive.release());
  if (*primitiveC == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_ERROR;
  }

  *output_size = 1;
  for (int i = 0; i < tf_op.input_size(); i++) {
    inputs->emplace_back(tf_op.input(i));
  }
  return RET_OK;
}
TFNodeRegistrar g_tfBatchNormParser("FusedBatchNormV3", new TFBatchNormParser());
TFNodeRegistrar g_tfFusedBatchNormParser("FusedBatchNorm", new TFBatchNormParser());
}  // namespace lite
}  // namespace mindspore
