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

#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateMergeParameter(const mindspore::lite::PrimitiveC *primitive) {
  OpParameter *merge_parameter = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (merge_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc Merge parameter failed.";
    return nullptr;
  }
  memset(merge_parameter, 0, sizeof(OpParameter));
  merge_parameter->type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(merge_parameter);
}
Registry MergeParameterRegistry(schema::PrimitiveType_Merge, PopulateMergeParameter);
}  // namespace lite
}  // namespace mindspore
