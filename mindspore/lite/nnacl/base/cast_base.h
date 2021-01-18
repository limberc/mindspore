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
#ifndef MINDSPORE_LITE_NNACL_CAST_BASE_H_
#define MINDSPORE_LITE_NNACL_CAST_BASE_H_

#include "nnacl/op_base.h"
#include "nnacl/nnacl_common.h"

#ifdef __cplusplus
extern "C" {
#endif

inline void BoolToFloat32(const bool *input, float *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float)input[i];
  }
}

inline void Uint8ToFloat32(const uint8_t *input, float *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float)input[i];
  }
}

inline void Int32ToFloat32(const int32_t *input, float *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float)input[i];
  }
}

inline void Fp16ToFloat32(const uint16_t *input, float *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = ShortToFloat32(input[i]);
  }
}

inline void Float32ToFp16(const float *input, uint16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = Float32ToShort(input[i]);
  }
}

inline void Float32ToInt32(const float *input, int32_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (int32_t)input[i];
  }
}

inline void Float32ToInt64(const float *input, int64_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (int64_t)input[i];
  }
}

inline void Int32ToInt64(const int32_t *input, int64_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (int64_t)input[i];
  }
}

inline void Float32ToInt16(const float *input, int16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (int16_t)input[i];
  }
}
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_CAST_BASE_H_
