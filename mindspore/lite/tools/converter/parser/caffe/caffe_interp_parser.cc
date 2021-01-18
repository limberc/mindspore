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

#include "tools/converter/parser/caffe/caffe_interp_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
PrimitiveC *CaffeInterpParser::ParseLitePrimitive(const caffe::LayerParameter &proto,
                                                  const caffe::LayerParameter &weight) {
  std::unique_ptr<schema::ResizeT> attr = std::make_unique<schema::ResizeT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  const caffe::InterpParameter &interpParam = proto.interp_param();
  if (interpParam.has_height()) {
    int64_t height = interpParam.height();
    if (height < 0) {
      MS_LOG(ERROR) << "Interp height must be > 0";
      return nullptr;
    }
    attr->newHeight = height;
  }

  if (interpParam.has_width()) {
    int64_t width = interpParam.width();
    if (width < 0) {
      MS_LOG(ERROR) << "Interp width must be > 0";
      return nullptr;
    }
    attr->newWidth = width;
  }
  attr->method = schema::ResizeMethod_LINEAR;
  attr->coordinateTransformMode = schema::CoordinateTransformMode_ALIGN_CORNERS;
  auto primitive = std::make_unique<schema::PrimitiveT>();
  primitive->value.type = schema::PrimitiveType_Resize;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

CaffeNodeRegistrar g_caffeInterpParser("Interp", new CaffeInterpParser());
}  // namespace lite
}  // namespace mindspore
