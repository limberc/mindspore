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

#include "tools/converter/quantizer/weight_quantizer.h"
#include <list>
#include <string>
#include <vector>
#include <unordered_map>
#include "src/common/common.h"

using std::string;
using std::vector;

namespace mindspore::lite::quant {
bool WeightQuantizer::IsPosNum(const std::string &str) {
  for (size_t i = 0; i < str.size(); i++) {
    if (str.at(i) < '0' || str.at(i) > '9') {
      return false;
    }
    if (str.at(i) == '0' && i == 0 && str.size() != 1) {
      return false;
    }
  }
  return true;
}

STATUS WeightQuantizer::WeightQuantInputCheck(const converter::Flags *config) {
  MS_ASSERT(config != nullptr);
  if (!WeightQuantizer::IsPosNum(config->quantWeightChannel)) {
    MS_LOG(ERROR) << "quantWeightChannel must be valid pos num.";
    return RET_ERROR;
  }
  if (!WeightQuantizer::IsPosNum(config->quantWeightSize)) {
    MS_LOG(ERROR) << "quantWeightSize must be valid pos num.";
    return RET_ERROR;
  }
  if (!WeightQuantizer::IsPosNum(config->bitNum)) {
    MS_LOG(ERROR) << "bitNum must be valid pos num.";
    return RET_ERROR;
  }
  int bitNum = std::stoi(config->bitNum);
  if (bitNum <= 0 || bitNum > 16) {
    MS_LOG(ERROR) << "bitNum should be more than 0 and less than 16 currently.";
    return RET_ERROR;
  }
  return RET_OK;
}

WeightQuantizer::WeightQuantizer(FuncGraphPtr graph, const PostQuantConfig &config) : Quantizer(graph) {
  quant_strategy_ = std::make_unique<QuantStrategy>(0, 0);
  config_param_ = config;
}

WeightQuantizer::WeightQuantizer(FuncGraphPtr graph, const std::string &config_file, const string &weightSize,
                                 const std::string &convWeightChannelThreshold, const std::string &bitNum)
    : Quantizer(graph) {
  this->config_file_ = config_file;
  auto quantSize = static_cast<size_t>(std::stoull(weightSize));
  this->bit_num_ = static_cast<size_t>(std::stoull(bitNum));
  auto convQuantWeightChannelThreshold = static_cast<size_t>(std::stoull(convWeightChannelThreshold));
  quant_strategy_ = std::make_unique<QuantStrategy>(quantSize, convQuantWeightChannelThreshold);
  quant_max_ = (1 << (unsigned int)(this->bit_num_ - 1)) - 1;
  quant_min_ = -(1 << (unsigned int)(this->bit_num_ - 1));
  // parse type_id_
  if (this->bit_num_ > 0 && this->bit_num_ <= 8) {
    type_id_ = kNumberTypeInt8;
  } else if (this->bit_num_ <= 16) {
    type_id_ = kNumberTypeInt16;
  } else {
    MS_LOG(ERROR) << "invalid input bits";
  }
}

WeightQuantizer::~WeightQuantizer() {
  for (const auto &fp32_output_tensor : fp32_output_tensors_) {
    for (const auto &kv : fp32_output_tensor) {
      delete kv.second;
    }
  }
}

STATUS WeightQuantizer::SetAbstract(ParamValueLitePtr param_value, ParameterPtr param_node,
                                    std::shared_ptr<PrimitiveC> primitive_c) {
  // set dtype
  param_value->set_tensor_type(type_id_);
  auto abstract_base = param_node->abstract();
  if (abstract_base == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << param_node->name();
    return RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract_base)) {
    MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << param_node->name();
    return RET_ERROR;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract_base);
  abstract_tensor->element()->set_type(TypeIdToType(type_id_));
  primitive_c->set_quant_type(schema::QuantType_WeightQuant);

  return RET_OK;
}

STATUS WeightQuantizer::DoConvQuantize(CNodePtr cnode) {
  auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr";
    return RET_ERROR;
  }

  auto input_node = cnode->input(2);
  if (!input_node->isa<Parameter>()) {
    return RET_ERROR;
  }

  ParameterPtr param_node;
  ParamValueLitePtr param_value;

  GetLiteParameter(input_node, &param_node, &param_value);
  if (param_node == nullptr || param_value == nullptr) {
    MS_LOG(ERROR) << "GetLiteParameter error";
    return RET_ERROR;
  }

  if (param_value->tensor_type() != mindspore::kNumberTypeFloat32) {
    MS_LOG(ERROR) << "model weight data type invalid which is " << param_value->tensor_type();
    return RET_ERROR;
  }
  auto status = RET_ERROR;
  if (type_id_ == kNumberTypeInt8) {
    status =
      QuantFilter<int8_t>(param_value, primitive_c, QuantType_WeightQuant, quant_max_, quant_min_, bit_num_, true);
  } else if (type_id_ == kNumberTypeInt16) {
    status =
      QuantFilter<int16_t>(param_value, primitive_c, QuantType_WeightQuant, quant_max_, quant_min_, bit_num_, true);
  }
  if (status != RET_OK) {
    MS_LOG(ERROR) << "QuantFilter failed : " << status;
    return status;
  }
  status = SetAbstract(param_value, param_node, primitive_c);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "SetAbstract failed : " << status;
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS WeightQuantizer::DoMulQuantize(CNodePtr cnode) {
  auto already_quant = false;
  ParamValueLitePtr param_value = nullptr;
  ParameterPtr param_node = nullptr;
  for (size_t i = 1; i < cnode->size(); i++) {
    auto inputNode = cnode->input(i);
    if (inputNode->isa<Parameter>()) {
      param_node = inputNode->cast<ParameterPtr>();
      if ((param_node != nullptr) && param_node->has_default()) {
        param_value = std::static_pointer_cast<ParamValueLite>(param_node->default_param());
        if ((param_value == nullptr) || (param_value->tensor_size() == 0) || (param_value->tensor_addr() == nullptr)) {
          param_value = nullptr;
          continue;
        } else if (param_value->tensor_type() == mindspore::kNumberTypeInt8 ||
                   param_value->tensor_type() == mindspore::kNumberTypeInt16) {
          MS_LOG(INFO) << "the node: " << cnode->fullname_with_scope() << " input_i: " << i << "has been "
                       << " quantized";
          already_quant = true;
          break;
        } else if (param_value->tensor_type() != mindspore::kNumberTypeFloat32) {
          param_value = nullptr;
          continue;
        } else {
          break;
        }
      }
    }
  }

  if (already_quant) {
    return RET_OK;
  }

  if (param_value == nullptr) {
    MS_LOG(ERROR) << "No valid input param node !";
    return RET_ERROR;
  }

  auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr";
    return RET_ERROR;
  }

  auto status = RET_ERROR;
  if (type_id_ == kNumberTypeInt8) {
    status =
      QuantFilter<int8_t>(param_value, primitive_c, QuantType_WeightQuant, quant_max_, quant_min_, bit_num_, true);
  } else if (type_id_ == kNumberTypeInt16) {
    status =
      QuantFilter<int16_t>(param_value, primitive_c, QuantType_WeightQuant, quant_max_, quant_min_, bit_num_, true);
  }
  if (status != RET_OK) {
    MS_LOG(ERROR) << "QuantFilter failed : " << status;
    return status;
  }
  status = SetAbstract(param_value, param_node, primitive_c);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "SetAbstract failed : " << status;
    return RET_ERROR;
  }

  return RET_OK;
}

STATUS WeightQuantizer::DoLstmQuntize(CNodePtr cnode) {
  MS_ASSERT(cnode != nullptr);
  auto op_name = cnode->fullname_with_scope();

  auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
  MS_ASSERT(primitive_c != nullptr);

  if (cnode->inputs().size() < 4) {
    MS_LOG(ERROR) << op_name << " inputs is " << cnode->inputs().size();
    return RET_ERROR;
  }
  {
    auto weight_i = cnode->input(2);
    ParameterPtr param_node;
    ParamValueLitePtr param_value;
    GetLiteParameter(weight_i, &param_node, &param_value);
    if (param_node == nullptr || param_value == nullptr) {
      MS_LOG(ERROR) << "GetLiteParameter error";
      return RET_ERROR;
    }
    if (param_value->tensor_type() != TypeId::kNumberTypeFloat32) {
      MS_LOG(WARNING) << "param_value tensor type is: " << param_value->tensor_type() << " not quant";
      return RET_OK;
    }
    if (param_value->tensor_size() / 4 < quant_strategy_->mWeightSize) {
      MS_LOG(INFO) << op_name << " weight_i cnt: " << param_value->tensor_size() / 4 << " < "
                   << quant_strategy_->mWeightSize;
      return RET_OK;
    }
    auto status = RET_ERROR;
    if (type_id_ == kNumberTypeInt8) {
      status =
        QuantFilter<int8_t>(param_value, primitive_c, QuantType_WeightQuant, quant_max_, quant_min_, bit_num_, false);
    } else if (type_id_ == kNumberTypeInt16) {
      status =
        QuantFilter<int16_t>(param_value, primitive_c, QuantType_WeightQuant, quant_max_, quant_min_, bit_num_, false);
    }
    if (status != RET_OK) {
      MS_LOG(ERROR) << "QuantFilter failed : " << status;
      return status;
    }
    status = SetAbstract(param_value, param_node, primitive_c);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "SetAbstract failed : " << status;
      return RET_ERROR;
    }
  }
  {
    auto weight_h = cnode->input(3);
    ParameterPtr param_node;
    ParamValueLitePtr param_value;
    GetLiteParameter(weight_h, &param_node, &param_value);
    if (param_node == nullptr || param_value == nullptr) {
      MS_LOG(ERROR) << "GetLiteParameter error";
      return RET_ERROR;
    }
    if (param_value->tensor_type() != TypeId::kNumberTypeFloat32) {
      MS_LOG(ERROR) << "param_value tensor type is: " << param_value->tensor_type() << " not quant";
      return RET_ERROR;
    }
    auto status = RET_ERROR;
    if (type_id_ == kNumberTypeInt8) {
      status = QuantFilter<int8_t>(param_value, primitive_c, QuantType_WeightQuant, quant_max_, quant_min_, bit_num_,
                                   false, 2);
    } else if (type_id_ == kNumberTypeInt16) {
      status = QuantFilter<int16_t>(param_value, primitive_c, QuantType_WeightQuant, quant_max_, quant_min_, bit_num_,
                                    false, 2);
    }
    if (status != RET_OK) {
      MS_LOG(ERROR) << "QuantFilter failed : " << status;
      return status;
    }
    status = SetAbstract(param_value, param_node, primitive_c);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "SetAbstract failed : " << status;
      return RET_ERROR;
    }
  }
  {
    if (cnode->inputs().size() > 4) {
      auto bias = cnode->input(4);
      ParameterPtr param_node;
      ParamValueLitePtr param_value;
      GetLiteParameter(bias, &param_node, &param_value);
      if (param_node == nullptr || param_value == nullptr) {
        MS_LOG(ERROR) << "GetLiteParameter error";
        return RET_ERROR;
      }
      if (param_value->tensor_type() != TypeId::kNumberTypeFloat32) {
        MS_LOG(ERROR) << "param_value tensor type is: " << param_value->tensor_type() << " not quant";
        return RET_ERROR;
      }
      auto status = RET_ERROR;
      if (type_id_ == kNumberTypeInt8) {
        status =
          QuantFilter<int8_t>(param_value, primitive_c, QuantType_WeightQuant, quant_max_, quant_min_, bit_num_, false);
      } else if (type_id_ == kNumberTypeInt16) {
        status = QuantFilter<int16_t>(param_value, primitive_c, QuantType_WeightQuant, quant_max_, quant_min_, bit_num_,
                                      false);
      }
      if (status != RET_OK) {
        MS_LOG(ERROR) << "QuantFilter failed : " << status;
        return status;
      }
      status = SetAbstract(param_value, param_node, primitive_c);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "SetAbstract failed : " << status;
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

STATUS WeightQuantizer::DoGatherQuntize(CNodePtr cnode) {
  auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
  MS_ASSERT(primitive_c != nullptr);

  auto weight_h = cnode->input(1);
  ParameterPtr param_node;
  ParamValueLitePtr param_value;
  GetLiteParameter(weight_h, &param_node, &param_value);
  if (param_node == nullptr || param_value == nullptr || param_value->tensor_type() != TypeId::kNumberTypeFloat32) {
    MS_LOG(INFO) << "This Gather op " << cnode->fullname_with_scope() << " can not quant weight";
    return RET_OK;
  }

  if (param_value->tensor_size() / 4 < quant_strategy_->mWeightSize) {
    MS_LOG(INFO) << cnode->fullname_with_scope() << " param cnt: " << param_value->tensor_size() / 4 << " < "
                 << quant_strategy_->mWeightSize;
    return RET_OK;
  }

  auto status = RET_ERROR;
  if (type_id_ == kNumberTypeInt8) {
    status =
      QuantFilter<int8_t>(param_value, primitive_c, QuantType_WeightQuant, quant_max_, quant_min_, bit_num_, false);
  } else if (type_id_ == kNumberTypeInt16) {
    status =
      QuantFilter<int16_t>(param_value, primitive_c, QuantType_WeightQuant, quant_max_, quant_min_, bit_num_, false);
  }
  if (status != RET_OK) {
    MS_LOG(ERROR) << "QuantFilter failed : " << status;
    return status;
  }
  status = SetAbstract(param_value, param_node, primitive_c);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "SetAbstract failed : " << status;
    return RET_ERROR;
  }
  return RET_OK;
}

constexpr float relative_tolerance = 1e-5;
constexpr float abs_tolerance = 1e-4;

template <typename T>
float CompareOutputData(const std::unordered_map<std::string, mindspore::tensor::MSTensor *> &expected_tensor,
                        const std::unordered_map<std::string, mindspore::tensor::MSTensor *> &compare_tensor) {
  auto valid_data = [](T data) -> bool { return (!std::isnan(data) && !std::isinf(data)); };

  float total_mean_error = 0.0f;
  int tensor_cnt = expected_tensor.size();

  if (tensor_cnt <= 0) {
    MS_LOG(ERROR) << "unexpected tensor_cnt: " << tensor_cnt;
    return RET_ERROR;
  }

  for (const auto &exp_tensor_pair : expected_tensor) {
    float mean_error = 0.0f;
    int error_cnt = 0;

    auto exp_tensor_name = exp_tensor_pair.first;
    auto exp_tensor = exp_tensor_pair.second;
    auto cmp_tensor_find_iter = compare_tensor.find(exp_tensor_name);
    if (cmp_tensor_find_iter == compare_tensor.end()) {
      MS_LOG(ERROR) << "can not find: " << exp_tensor_name;
      return RET_ERROR;
    }
    auto cmp_tensor = cmp_tensor_find_iter->second;

    auto exp_tensor_shape = exp_tensor->shape();
    auto cmp_tensor_shape = cmp_tensor->shape();
    if (exp_tensor_shape != cmp_tensor_shape) {
      MS_LOG(ERROR) << "exp tensor shape not equal to cmp. exp_tensor_elem_cnt: " << exp_tensor->ElementsNum()
                    << " cmp_tensor_elem_cnt: " << cmp_tensor->ElementsNum();
      return RET_ERROR;
    }
    auto exp_data = static_cast<T *>(exp_tensor->MutableData());
    auto cmp_data = static_cast<T *>(cmp_tensor->MutableData());
    auto elem_cnt = exp_tensor->ElementsNum();
    for (int i = 0; i < elem_cnt; i++) {
      if (!valid_data(exp_data[i]) || !valid_data(cmp_data[i])) {
        MS_LOG(ERROR) << "data is not valid. exp: " << exp_data[i] << " cmp: " << cmp_data[i] << " index: " << i;
        return RET_ERROR;
      }
      auto tolerance = abs_tolerance + relative_tolerance * fabs(exp_data[i]);
      auto abs_error = std::fabs(exp_data[i] - cmp_data[i]);
      if (abs_error > tolerance) {
        if (fabs(exp_data[i] == 0)) {
          if (abs_error > 1e-5) {
            mean_error += abs_error;
            error_cnt++;
          } else {
            // it is ok, very close to 0
            continue;
          }
        } else {
          mean_error += abs_error / (fabs(exp_data[i]) + FLT_MIN);
          error_cnt++;
        }
      } else {
        // it is ok, no error
        continue;
      }
    }  // end one tensor data loop
    total_mean_error += mean_error / elem_cnt;
  }  // end tensor loop
  return total_mean_error / tensor_cnt;
}

STATUS WeightQuantizer::RunFp32Graph(FuncGraphPtr func_graph) {
  auto image_cnt = images_.at(0).size();
  if (!config_param_.input_shapes.empty()) {
    if (config_param_.input_shapes.size() != image_cnt) {
      MS_LOG(ERROR) << "input_shapes size: " << config_param_.input_shapes.size() << " image_cnt: " << image_cnt;
      return RET_ERROR;
    }
  }
  // 0.1 Create Fp32 Session
  flags.quantType = schema::QuantType_QUANT_NONE;
  auto fp32_sm = CreateSessionByFuncGraph(func_graph, flags, config_param_.thread_num);
  auto fp32_session = fp32_sm.session;
  auto fp32_model = fp32_sm.model;
  if (fp32_session == nullptr || fp32_model == nullptr) {
    MS_LOG(ERROR) << "CreateSessoin fail";
    delete fp32_model;
    return RET_ERROR;
  }
  auto fp32_inputs = fp32_session->GetInputs();
  fp32_output_tensors_.resize(image_cnt);
  // 0.3 save fp32 output
  for (size_t i = 0; i < image_cnt; i++) {
    if (!config_param_.input_shapes.empty()) {
      auto status = fp32_session->Resize(fp32_inputs, {config_param_.input_shapes[i]});
      if (status != RET_OK) {
        MS_LOG(ERROR) << "session Resize fail";
        delete fp32_sm.session;
        delete fp32_sm.model;
        return RET_ERROR;
      }
    }
    for (size_t input_index = 0; input_index < fp32_inputs.size(); input_index++) {
      auto status = CopyInputDataToTensor(input_index, i, images_, fp32_inputs[input_index]);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "generate input data from images failed!";
        delete fp32_sm.session;
        delete fp32_sm.model;
        return RET_ERROR;
      }
    }
    auto status = fp32_session->RunGraph();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "RunGraph fail";
      delete fp32_sm.session;
      delete fp32_sm.model;
      return RET_ERROR;
    }
    auto fp32_outputs = fp32_session->GetOutputs();
    for (const auto &kv : fp32_outputs) {
      auto *tensor = kv.second;
      auto *lite_tensor = reinterpret_cast<lite::Tensor *>(tensor);
      if (lite_tensor == nullptr) {
        MS_LOG(ERROR) << "not lite tensor";
        delete fp32_sm.session;
        delete fp32_sm.model;
        return RET_ERROR;
      }
      auto *new_tensor = Tensor::CopyTensor(*lite_tensor, true);
      fp32_output_tensors_[i][kv.first] = new_tensor;
    }
  }
  delete fp32_sm.session;
  delete fp32_sm.model;
  return RET_OK;
}

STATUS WeightQuantizer::DoMiexedQuant(FuncGraphPtr func_graph) {
  // 0.2 Parse input calib files
  auto status = CollectCalibInputs(config_param_.image_paths, config_param_.batch_count, &images_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "CollectCalibInputs fail";
    return RET_ERROR;
  }

  MS_LOG(DEBUG) << "run fp32 model";
  status = RunFp32Graph(func_graph);
  if (status != RET_OK) {
    return RET_ERROR;
  }

  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    auto op_type = NodePrimitiveType(cnode);
    if (op_type == schema::PrimitiveType_Lstm) {
      status = DoLstmQuntize(cnode);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "DoLstmQuntize error";
        return RET_ERROR;
      }
    } else if (op_type == schema::PrimitiveType_Gather) {
      status = DoGatherQuntize(cnode);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "DoGatherQuntize error";
        return RET_ERROR;
      }
    }
  }
  auto image_cnt = images_.at(0).size();
  if (!config_param_.input_shapes.empty()) {
    if (config_param_.input_shapes.size() != image_cnt) {
      MS_LOG(ERROR) << "input_shapes size: " << config_param_.input_shapes.size() << " image_cnt: " << image_cnt;
      return RET_ERROR;
    }
  }

  for (auto iter = cnodes.end(); iter != cnodes.begin();) {
    auto cnode = *(--iter);
    auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "primitive_c is null.";
      return RET_ERROR;
    }
    auto op_name = cnode->fullname_with_scope();
    MS_LOG(DEBUG) << "process node: " << op_name
                  << " type: " << schema::EnumNamePrimitiveType((schema::PrimitiveType)primitive_c->Type());
    if (quant_strategy_->CanConvOpQuantized(cnode) || quant_strategy_->CanMulOpQuantized(cnode)) {
      auto input_node = cnode->input(2);
      if (!input_node->isa<Parameter>()) {
        MS_LOG(WARNING) << op_name << " the second input is not parameter";
        continue;
      }
      auto param_node = input_node->cast<ParameterPtr>();
      if (!param_node->has_default()) {
        MS_LOG(WARNING) << op_name << " the second input can not convert to parameter";
        continue;
      }
      auto param_value = std::static_pointer_cast<ParamValueLite>(param_node->default_param());
      if (param_value == nullptr) {
        MS_LOG(WARNING) << op_name << " the second input can not convert to parameter";
        continue;
      }
      if (param_value->tensor_type() != TypeId::kNumberTypeFloat32) {
        MS_LOG(WARNING) << op_name << " the second input type is not float";
        continue;
      }
      // copy origin data in case to recover
      auto *raw_data = static_cast<float *>(param_value->tensor_addr());
      auto elem_count = param_value->tensor_shape_size();
      auto origin_data = malloc(sizeof(float) * elem_count);
      auto ret = memcpy_s(origin_data, sizeof(float) * elem_count, raw_data, param_value->tensor_size());
      if (ret != EOK) {
        MS_LOG(ERROR) << "memcpy fail: "
                      << " dst size: " << sizeof(float) * elem_count << " src size: " << param_value->tensor_size();
        return RET_ERROR;
      }
      // 1. try quant
      for (int bit_num_t = 2; bit_num_t <= 8; bit_num_t++) {
        type_id_ = TypeId::kNumberTypeInt8;
        int quant_max_t = (1 << (unsigned int)(bit_num_t - 1)) - 1;
        int quant_min_t = -(1 << (unsigned int)(bit_num_t - 1));

        if (type_id_ == TypeId::kNumberTypeInt8) {
          status = QuantFilter<int8_t>(param_value, primitive_c, QuantType::QuantType_WeightQuant, quant_max_t,
                                       quant_min_t, bit_num_t, true);
        } else if (type_id_ == TypeId::kNumberTypeInt16) {
          status = QuantFilter<int16_t>(param_value, primitive_c, QuantType::QuantType_WeightQuant, quant_max_t,
                                        quant_min_t, bit_num_t, true);
        } else {
          MS_LOG(ERROR) << "unexpected type_id_: " << type_id_;
          return RET_ERROR;
        }
        if (status != RET_OK) {
          MS_LOG(ERROR) << "quant filter fail.";
          return RET_ERROR;
        }
        status = SetAbstract(param_value, param_node, primitive_c);
        if (status != RET_OK) {
          MS_LOG(ERROR) << "SetAbstract failed : " << status;
          return RET_ERROR;
        }
        // 2. evaluate the quant
        // 2.1 create quant session, get input, output tensor
        flags.quantType = schema::QuantType_WeightQuant;
        auto quant_sm = CreateSessionByFuncGraph(func_graph, flags, config_param_.thread_num);
        auto quant_session = std::unique_ptr<session::LiteSession>(quant_sm.session);
        if (quant_session == nullptr) {
          MS_LOG(ERROR) << "create session error: " << status;
          delete quant_sm.model;
          return RET_ERROR;
        }
        auto quant_inputs = quant_session->GetInputs();

        auto mean_error = 0.0f;
        for (size_t i = 0; i < image_cnt; i++) {
          if (!config_param_.input_shapes.empty()) {
            status = quant_session->Resize(quant_inputs, {config_param_.input_shapes[i]});
            if (status != RET_OK) {
              MS_LOG(ERROR) << "session Resize fail";
              delete quant_sm.model;
              return RET_ERROR;
            }
          }

          // set multi-input data
          for (size_t input_index = 0; input_index < quant_inputs.size(); input_index++) {
            status = CopyInputDataToTensor(input_index, i, images_, quant_inputs[input_index]);
            if (status != RET_OK) {
              MS_LOG(ERROR) << "generate input data from images failed!";
              delete quant_sm.model;
              return RET_ERROR;
            }
          }
          status = quant_session->RunGraph();
          if (status != RET_OK) {
            MS_LOG(ERROR) << "quant session run error";
            delete quant_sm.model;
            return RET_ERROR;
          }
          // 3. compare betwen quant and fp32
          auto quant_outputs = quant_session->GetOutputs();
          mean_error += CompareOutputData<float>(fp32_output_tensors_[i], quant_outputs);
        }  // end_for: calib data loop
        delete quant_sm.model;
        mean_error = mean_error / image_cnt;
        if (mean_error <= config_param_.mean_error_threshold) {
          MS_LOG(DEBUG) << "op: " << op_name << " got mixed bit: " << bit_num_t << " mean_error: " << mean_error;
          opname_bit_[op_name] = bit_num_t;
          break;
        } else if (bit_num_t != 8) {
          MS_LOG(DEBUG) << "op: " << op_name << " intermediate bit: " << bit_num_t << " mean_error: " << mean_error
                        << " [recover]";
          // recover
          status = UpdateTensorDataAndSize(param_value, origin_data, sizeof(float) * elem_count);
          if (status != RET_OK) {
            MS_LOG(ERROR) << "UpdateTensorDataAndSize fail";
            return RET_ERROR;
          }
        } else {
          MS_LOG(DEBUG) << "op: " << op_name << " set bit: " << bit_num_t << " mean_error: " << mean_error;
          opname_bit_[op_name] = bit_num_t;
        }
      }  // end bit loop
      free(origin_data);
    }  //  if: conv and matmul
  }    // end loop: all cnode
  for (const auto &kv : opname_bit_) {
    MS_LOG(INFO) << "op: " << kv.first << " bit:" << kv.second;
  }
  return RET_OK;
}

STATUS WeightQuantizer::DoFixedQuant(FuncGraphPtr func_graph) {
  MS_ASSERT(func_graph != nullptr);
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "primitive_c is nullptr";
      return RET_ERROR;
    }
    auto op_name = cnode->fullname_with_scope();
    auto op_type = (schema::PrimitiveType)primitive_c->Type();

    if (quant_strategy_->CanConvOpQuantized(cnode)) {
      auto status = DoConvQuantize(cnode);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "DoConvQuantize error";
        return RET_ERROR;
      }
    } else if (quant_strategy_->CanMulOpQuantized(cnode)) {
      auto status = DoMulQuantize(cnode);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "DoMulQuantize error";
        return RET_ERROR;
      }
    } else if (op_type == schema::PrimitiveType_Lstm) {
      auto status = DoLstmQuntize(cnode);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "DoLstmQuntize error";
        return RET_ERROR;
      }
    } else if (op_type == schema::PrimitiveType_Gather) {
      auto status = DoGatherQuntize(cnode);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "DoGatherQuntize error";
        return RET_ERROR;
      }
    } else {
      MS_LOG(DEBUG) << op_name << " of type: " << schema::EnumNamePrimitiveType(op_type) << " no need quant";
    }
  }
  return RET_OK;
}

STATUS WeightQuantizer::DoQuantize(FuncGraphPtr func_graph) {
  MS_ASSERT(func_graph != nullptr);

  if (!config_file_.empty()) {
    auto ret = ParseConfigFile(config_file_, &config_param_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ReadConfig error.";
      return RET_ERROR;
    }
  }

  if (config_param_.mixed) {
    bit_num_ = 8;
    quant_max_ = (1 << (unsigned int)(this->bit_num_ - 1)) - 1;
    quant_min_ = -(1 << (unsigned int)(this->bit_num_ - 1));
    type_id_ = kNumberTypeInt8;
    MS_LOG(INFO) << "Do mixed bit quantization";
    return DoMiexedQuant(func_graph);
  }

  return DoFixedQuant(func_graph);
}
}  // namespace mindspore::lite::quant
