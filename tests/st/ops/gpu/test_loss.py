# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" test loss """
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.nn.loss.loss import _Loss
from mindspore.nn.loss.loss import L1Loss
import mindspore.context as context

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

class WeightedLoss(_Loss):
    def __init__(self, reduction='mean', weights=1.0):
        super(WeightedLoss, self).__init__(reduction)
        self.abs = P.Abs()
        self.weights = weights

    def construct(self, base, target):
        x = self.abs(base - target)
        return self.get_loss(x, self.weights)

def test_WeightedLoss():
    loss = WeightedLoss()
    input_data = Tensor(np.array([[1, 2, 3], [2, 3, 4]]).astype(np.float32))
    target_data = Tensor(np.array([[0, 2, 5], [3, 1, 1]]).astype(np.float32))
    output_data = loss(input_data, target_data)

    error_range = np.ones(shape=output_data.shape) * 10e-6
    loss = WeightedLoss(weights=2.0)
    test_output = loss(input_data, target_data)
    diff = test_output - output_data * 2.0
    assert np.all(abs(diff.asnumpy()) < error_range)

    loss = WeightedLoss(weights=3)
    test_output = loss(input_data, target_data)
    diff = test_output - output_data * 3
    assert np.all(abs(diff.asnumpy()) < error_range)

    loss = WeightedLoss(weights=Tensor(np.array([[0.7, 0.3], [0.7, 0.3]])))
    y_true = Tensor(np.array([[0., 1.], [0., 0.]]), mindspore.float32)
    y_pred = Tensor(np.array([[1., 1.], [1., 0.]]), mindspore.float32)
    test_data = 0.35
    output = loss(y_true, y_pred)
    diff = test_data - output.asnumpy()
    assert np.all(abs(diff) < error_range)

class CustomLoss(_Loss):
    def __init__(self, reduction='mean'):
        super(CustomLoss, self).__init__(reduction)
        self.abs = P.Abs()

    def construct(self, base, target):
        x = self.abs(base - target)
        return self.get_loss(x, weights=2.0)

def test_CustomLoss():
    loss = L1Loss()
    input_data = Tensor(np.array([[1, 2, 3], [2, 3, 4]]).astype(np.float32))
    target_data = Tensor(np.array([[0, 2, 5], [3, 1, 1]]).astype(np.float32))
    output_data = loss(input_data, target_data)

    error_range = np.ones(shape=output_data.shape) * 10e-6
    customloss = CustomLoss()
    test_output = customloss(input_data, target_data)
    diff = test_output - output_data * 2.0
    assert np.all(abs(diff.asnumpy()) < error_range)
