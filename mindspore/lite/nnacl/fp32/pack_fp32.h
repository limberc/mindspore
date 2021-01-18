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
#ifndef MINDSPORE_LITE_NNACL_FP32_PACK_H_
#define MINDSPORE_LITE_NNACL_FP32_PACK_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/conv_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

void PackHWCToWHC(const float *src, float *dst, int height, int width, int channel);
void PackNHWCToNC4HW4Fp32(const void *src, void *dst, int batch, int plane, int channel);
void PackNCHWToNC4HW4Fp32(const void *src, void *dst, int batch, int plane, int channel);
void PackNHWCToNHWC4Fp32(const void *src, void *dst, int batch, int plane, int channel);
void PackNHWCToNHWC8Fp32(const void *src, void *dst, int batch, int plane, int channel);
void PackNHWCToNCHWFp32(const void *src, void *dst, int batch, int plane, int channel);
void PackNCHWToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel);
void PackNHWC4ToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel);
void PackNC4HW4ToNHWC4Fp32(const void *src, void *dst, int batch, int plane, int channel);
void PackNC4HW4ToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel);
void PackNHWCToC8HWN8Fp32(const void *src, void *dst, int batch, int plane, int channel);

void PackWeightKHWToHWKFp32(const void *src, void *dst, int plane, int channel);
void PackDepthwiseIndirectWeightC4Fp32(const void *src, void *dst, int height, int width, int channel);
void PackDepthwiseIndirectWeightC8Fp32(const void *src, void *dst, int height, int width, int channel);
void Im2ColPackUnitFp32(const float *input_data, const ConvParameter *conv_param, float *packed_input, int real_cal_num,
                        int block_index);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP32_PAD_H_
