/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "tools/converter/anf_transform.h"
#include <memory>
#include <string>
#include "src/common/log_adapter.h"
#include "tools/optimizer/fusion/conv_biasadd_fusion.h"
#include "tools/optimizer/fusion/conv_activation_fusion.h"
#include "tools/optimizer/fusion/conv_tuple_activation_fusion.h"
#include "tools/optimizer/fusion/conv_scale_fusion.h"
#include "tools/optimizer/fusion/conv_bn_fusion.h"
#include "tools/optimizer/fusion/conv_tuplegetitem_fusion.h"
#include "tools/optimizer/fusion/constant_folding_fusion.h"
#include "tools/optimizer/fusion/layer_norm_fusion.h"
#include "tools/optimizer/fusion/batchmatmul_fusion.h"
#include "tools/optimizer/fusion/sigmoid_mul_fusion.h"
#include "tools/optimizer/fusion/conv_conv_fusion.h"
#include "tools/optimizer/fusion/tflite_lstm_cell_fusion.h"
#include "tools/optimizer/fusion/tf_lstm_cell_fusion.h"
#include "tools/optimizer/fusion/bidirection_tf_gru_cell_fusion.h"
#include "tools/optimizer/graph/mindir_adjust_pass.h"
#include "tools/optimizer/graph/mindir_inputs_adjust_pass.h"
#include "tools/optimizer/graph/identity_remove_pass.h"
#include "tools/optimizer/graph/weight_format_hardcode_pass.h"
#include "tools/optimizer/graph/weight_format_transform_pass.h"
#include "tools/optimizer/graph/clip_convert_activation_pass.h"
#include "tools/optimizer/graph/group_depthwise_op_convert_pass.h"
#include "tools/optimizer/graph/tflite_inputs_order_exchange_pass.h"
#include "tools/optimizer/graph/onnx_inputs_adjust_pass.h"
#include "tools/optimizer/graph/update_conv2d_param_pass.h"
#include "tools/optimizer/graph/unused_cast_node_remove_pass.h"
#include "tools/optimizer/graph/unused_transpose_node_remove_pass.h"
#include "tools/optimizer/graph/infershape_pass.h"
#include "tools/optimizer/graph/slice_prepose_pass.h"
#include "tools/optimizer/graph/while_pass.h"
#include "tools/optimizer/graph/if_pass.h"
#include "tools/converter/quantizer/post_training_quantizer.h"
#include "tools/converter/quantizer/quant_cast.h"
#include "tools/converter/quantizer/weight_quantizer.h"

using std::string;
namespace mindspore::lite {
AnfTransform::AnfTransform() = default;

AnfTransform::~AnfTransform() = default;

FuncGraphPtr AnfTransform::TransformSingleFuncGraph(const FuncGraphPtr &old_graph, const converter::Flags *config) {
  MS_ASSERT(nullptr != old_graph);
  if (config == nullptr) {
    MS_LOG(ERROR) << "config should be specified";
    return nullptr;
  }
  if (old_graph->has_flag("HasTransformed")) {
    old_graph->set_flag("HasTransformed", false);
    return old_graph;
  }
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto fusion_pm = std::make_shared<opt::PassManager>("anf fusion pass manager", false);
  auto graph_pm = std::make_shared<opt::PassManager>("anf graph pass manager", true);
  auto convert_pm = std::make_shared<opt::PassManager>("anf graph convert pass manager", true);

  if (config->fmk == converter::FmkType_MS) {
    auto mindir_adjust_pass = std::make_shared<opt::MindirAdjustPass>();
    mindir_adjust_pass->SetFmkType(config->fmk);
    mindir_adjust_pass->SetQuantType(config->quantType);
    if (!mindir_adjust_pass->Run(old_graph)) {
      MS_LOG(ERROR) << "mindir adjust failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
      return nullptr;
    }
    auto mindir_inputs_adjust_pass = std::make_shared<opt::MindirInputAdjustOpPass>();
    if (!mindir_inputs_adjust_pass->Run(old_graph)) {
      MS_LOG(ERROR) << "mindir inputs adjust failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
      return nullptr;
    }
  }

  // onnx pre adjustment
  if (config->fmk == converter::FmkType_ONNX) {
    auto onnx_adjust_pass = std::make_shared<opt::OnnxInputAdjustOpPass>();
    if (!onnx_adjust_pass->Run(old_graph)) {
      MS_LOG(ERROR) << "onnx adjust failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
      return nullptr;
    }
  }

  if (config->fmk == lite::converter::FmkType_TFLITE || config->fmk == lite::converter::FmkType_TF ||
      config->fmk == lite::converter::FmkType_ONNX) {
    graph_pm->AddPass(std::make_shared<opt::WhilePass>());
    graph_pm->AddPass(std::make_shared<opt::IfPass>());
  }

  // for now - training is not supporting fuse operations
  if (!config->trainModel) {
    // remove quantdtype when awaretraining
    fusion_pm->AddPass(std::make_shared<opt::RemoveIdentityOpPass>());
    fusion_pm->AddPass(std::make_shared<opt::ConvBiasaddFusion>());
    auto conv_bn_pass = std::make_shared<opt::ConvBatchNormFusion>();
    conv_bn_pass->SetFmkType(config->fmk);
    fusion_pm->AddPass(conv_bn_pass);
    auto conv_scale_pass = std::make_shared<opt::ConvScaleFusion>();
    conv_scale_pass->SetFmkType(config->fmk);
    fusion_pm->AddPass(conv_scale_pass);
    fusion_pm->AddPass(std::make_shared<opt::LayerNormFusion>());
    fusion_pm->AddPass(std::make_shared<opt::BatchMatMulFusion>());
    fusion_pm->AddPass(std::make_shared<opt::SigmoidMulFusion>());
    fusion_pm->AddPass(std::make_shared<opt::ConvActivationFusion>());
    fusion_pm->AddPass(std::make_shared<opt::ConvTupleGetItemFusion>());
    fusion_pm->AddPass(std::make_shared<opt::ConvTupleActivationFusion>());
    fusion_pm->AddPass(std::make_shared<opt::TfliteLstmCellFusion>());
    fusion_pm->AddPass(std::make_shared<opt::TfLstmCellFusion>());
    fusion_pm->AddPass(std::make_shared<opt::BiDirectionTfGruCellFusion>());
  }
  auto weight_format_hardcode_pass = std::make_shared<opt::WeightFormatHardCodePass>();
  weight_format_hardcode_pass->SetFmkType(config->fmk);
  weight_format_hardcode_pass->SetQuantType(config->quantType);
  graph_pm->AddPass(weight_format_hardcode_pass);
  auto weight_format_transform_pass = std::make_shared<opt::WeightFormatTransformPass>();
  weight_format_transform_pass->SetFmkType(config->fmk);
  weight_format_transform_pass->SetQuantType(config->quantType);
  graph_pm->AddPass(weight_format_transform_pass);
  auto infershape_pass = std::make_shared<opt::InferShapePass>();
  infershape_pass->SetFmkType(config->fmk);
  graph_pm->AddPass(infershape_pass);
  auto slice_prepose_pass = std::make_shared<opt::SlicePreposePass>();
  slice_prepose_pass->SetFmkType(config->fmk);
  graph_pm->AddPass(slice_prepose_pass);

  if (config->fmk == lite::converter::FmkType_MS) {
    auto remove_unused_cast_pass = std::make_shared<opt::RemoveUnusedCastOpPass>();
    if (remove_unused_cast_pass == nullptr) {
      MS_LOG(ERROR) << "RemoveUnusedCastOpPass shoud be specified";
      return nullptr;
    }
    remove_unused_cast_pass->SetFmkType(config->fmk);
    fusion_pm->AddPass(remove_unused_cast_pass);
  }
  if (config->fmk == lite::converter::FmkType_ONNX) {
    auto remove_unused_transpose_pass = std::make_shared<opt::RemoveUnusedTransposeOpPass>();
    if (remove_unused_transpose_pass == nullptr) {
      MS_LOG(ERROR) << "RemoveUnusedTransposeOpPass shoud be specified";
      return nullptr;
    }
    remove_unused_transpose_pass->SetFmkType(config->fmk);
    fusion_pm->AddPass(remove_unused_transpose_pass);
  }
  auto const_fold_pm = std::make_shared<opt::PassManager>("const fold fusion pass manager", false);
  if (!config->trainModel) {
    auto inne_context_ptr = std::make_shared<lite::InnerContext>();
    inne_context_ptr->Init();
    const_fold_pm->AddPass(std::make_shared<opt::ConstFoldPass>(inne_context_ptr));
  }
  auto update_conv2d_param_pass = std::make_shared<opt::UpdateConv2DParamPass>();
  update_conv2d_param_pass->SetFmkType(config->fmk);
  const_fold_pm->AddPass(update_conv2d_param_pass);
  fusion_pm->AddPass(std::make_shared<opt::ConvConvFusion>());
  convert_pm->AddPass(std::make_shared<opt::ClipConvertActivationPass>());
  if (config->fmk == lite::converter::FmkType_TFLITE) {
    convert_pm->AddPass(std::make_shared<opt::GroupDepthwiseOpConvertPass>());
    convert_pm->AddPass(std::make_shared<opt::TfliteInputsOrderExchangePass>());
  }
  optimizer->AddPassManager(const_fold_pm);
  optimizer->AddPassManager(convert_pm);
  optimizer->AddPassManager(fusion_pm);
  optimizer->AddPassManager(graph_pm);
  auto new_graph = optimizer->Optimize(old_graph);
  if (new_graph == nullptr) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_NULL_PTR);
    return nullptr;
  }
  // quant
  if (config->quantType == schema::QuantType_PostTraining) {
    if (!quant::WeightQuantizer::IsPosNum(config->bitNum)) {
      MS_LOG(ERROR) << "bitNum must be valid pos num.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
      return nullptr;
    }
    this->mQuantizer =
      std::make_unique<quant::PostTrainingQuantizer>(new_graph, config->configFile, std::stoi(config->bitNum));
    if (mQuantizer == nullptr) {
      MS_LOG(ERROR) << "New PostTrainingQuantizer failed";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_MEMORY_FAILED);
      return nullptr;
    }
  } else if (config->quantType == schema::QuantType_WeightQuant) {
    if (quant::WeightQuantizer::WeightQuantInputCheck(config) != RET_OK) {
      MS_LOG(ERROR) << "weight quant input param error";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
      return nullptr;
    }
    this->mQuantizer = std::make_unique<quant::WeightQuantizer>(new_graph, config->configFile, config->quantWeightSize,
                                                                config->quantWeightChannel, config->bitNum);
    if (mQuantizer == nullptr) {
      MS_LOG(ERROR) << "New WeightQuantizer failed";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_MEMORY_FAILED);
      return nullptr;
    }
  }
  if (mQuantizer != nullptr) {
    mQuantizer->flags = *config;
    auto status = mQuantizer->DoQuantize(new_graph);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Quant failed " << status;
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
      return nullptr;
    }
  }
  return new_graph;
}

STATUS AnfTransform::GetAllFuncGraph(const FuncGraphPtr &main_graph, FuncGraphPtrList *subgraphs,
                                     std::vector<ValueNodePtr> *vnodes) {
  auto nodes = TopoSort(main_graph->get_return());
  for (auto &node : nodes) {
    auto fg = GetValueNode<FuncGraphPtr>(node);
    if (fg) {
      vnodes->push_back(utils::cast<ValueNodePtr>(node));
      subgraphs->push_back(fg);
    }
  }
  return RET_OK;
}

FuncGraphPtr AnfTransform::Transform(const FuncGraphPtr &main_graph, const converter::Flags *config) {
  // transform main_graph
  auto new_main_graph = TransformSingleFuncGraph(main_graph, config);
  if (new_main_graph == nullptr) {
    MS_LOG(ERROR) << "TransformSingleFuncGraph failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return nullptr;
  }

  // transform sub_graph
  FuncGraphPtrList subgraphs{};
  std::vector<ValueNodePtr> vnodes{};
  int ret = GetAllFuncGraph(main_graph, &subgraphs, &vnodes);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GetAllFuncGraph failed " << ret;
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return nullptr;
  }
  for (size_t i = 0; i < subgraphs.size(); i++) {
    auto new_graph = Transform(subgraphs.at(i), config);
    new_graph->set_flag("HasTransformed", true);
    vnodes.at(i)->set_value(new_graph);
  }

  return new_main_graph;
}
}  // namespace mindspore::lite
