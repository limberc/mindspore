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
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateWhereParameter(const mindspore::lite::PrimitiveC *primitive) {
  OpParameter *where_parameter = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (where_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc Where parameter failed.";
    return nullptr;
  }
  memset(where_parameter, 0, sizeof(OpParameter));
  where_parameter->type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(where_parameter);
}
Registry WhereParameterRegistry(schema::PrimitiveType_Where, PopulateWhereParameter);
}  // namespace lite
}  // namespace mindspore
