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
#include "runtime/device/ascend/profiling/profiling_callback_register.h"

VMCallbackRegister &VMCallbackRegister::GetInstance() {
  static VMCallbackRegister instance;
  return instance;
}

bool VMCallbackRegister::Registe(Status (*pRegProfCtrlCallback)(MsprofCtrlCallback),
                                 Status (*pRegProfSetDeviceCallback)(MsprofSetDeviceCallback),
                                 Status (*pRegProfReporterCallback)(MsprofReporterCallback),
                                 Status (*pProfCommandHandle)(ProfCommandHandleType, void *, uint32_t)) {
  return false;
}

void VMCallbackRegister::ForceMsprofilerInit() {}
