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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_CONVOLUTION_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_CONVOLUTION_PARSER_H_

#include <vector>
#include "tools/converter/parser/caffe/caffe_node_parser.h"
#include "tools/converter/parser/caffe/caffe_node_parser_registry.h"
#include "tools/converter/parser/caffe/caffe_conv_base_parser.h"

namespace mindspore {
namespace lite {
class CaffeConvolutionParser : public CaffeNodeParser {
 public:
  CaffeConvolutionParser() : CaffeNodeParser("convolution") {}
  ~CaffeConvolutionParser() override = default;

  STATUS Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight, schema::CNodeT *op,
               std::vector<schema::TensorT *> *weightVec) override;

 private:
  static STATUS ParseGroupConvolution(schema::CNodeT *op, schema::Conv2DT *attr);
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_CONVOLUTION_PARSER_H_
