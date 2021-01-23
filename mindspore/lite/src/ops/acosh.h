//
// Created by Limber Cheng on 2021/1/23.
//
#ifndef LITE_MINDSPORE_LITE_C_OPS_ACOSH_H_
#define LITE_MINDSPORE_LITE_C_OPS_ACOSH_H_

#include <vector>
#include <set>
#include <cmath>

#include "src/ops/arithmetic_self.h"

namespace mindspore {
namespace lite {
class ACosh : public ArithmeticSelf {
 public:
  ACosh() = default;
  ~ACosh() = default;
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(ACosh, ArithmeticSelf);
  explicit ACosh(schema::PrimitiveT *primitive) : ArithmeticSelf(primitive) {}
#else
  int UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) override;
#endif
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_ACOSH_H_
