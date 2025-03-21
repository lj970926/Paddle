# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# repo: PaddleDetection
# model: configs^ppyolo^ppyolov2_r50vd_dcn_365e_coco_single_dy2st_train
# api:paddle.tensor.math.maximum||api:paddle.tensor.math.maximum||api:paddle.tensor.math.minimum||api:paddle.tensor.math.minimum||method:__sub__||method:clip||method:__sub__||method:clip||method:__mul__||method:__sub__||method:__sub__||method:__mul__||method:clip||method:__sub__||method:__sub__||method:__mul__||method:clip||method:__add__||method:__sub__||method:__add__||method:__truediv__||api:paddle.nn.functional.loss.binary_cross_entropy_with_logits||method:__mul__
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1, 3, 96, 96, 1], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1, 3, 96, 96, 1], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [1, 3, 96, 96, 1], dtype: paddle.float32, stop_gradient: False)
        var_3,  # (shape: [1, 3, 96, 96, 1], dtype: paddle.float32, stop_gradient: False)
        var_4,  # (shape: [1, 3, 96, 96, 1], dtype: paddle.float32, stop_gradient: False)
        var_5,  # (shape: [1, 3, 96, 96, 1], dtype: paddle.float32, stop_gradient: True)
        var_6,  # (shape: [1, 3, 96, 96, 1], dtype: paddle.float32, stop_gradient: True)
        var_7,  # (shape: [1, 3, 96, 96, 1], dtype: paddle.float32, stop_gradient: True)
        var_8,  # (shape: [1, 3, 96, 96, 1], dtype: paddle.float32, stop_gradient: True)
    ):
        var_9 = paddle.tensor.math.maximum(var_1, var_5)
        var_10 = paddle.tensor.math.maximum(var_2, var_6)
        var_11 = paddle.tensor.math.minimum(var_3, var_7)
        var_12 = paddle.tensor.math.minimum(var_4, var_8)
        var_13 = var_11 - var_9
        var_14 = var_13.clip(0)
        var_15 = var_12 - var_10
        var_16 = var_15.clip(0)
        var_17 = var_14 * var_16
        var_18 = var_3 - var_1
        var_19 = var_4 - var_2
        var_20 = var_18 * var_19
        var_21 = var_20.clip(0)
        var_22 = var_7 - var_5
        var_23 = var_8 - var_6
        var_24 = var_22 * var_23
        var_25 = var_24.clip(0)
        var_26 = var_21 + var_25
        var_27 = var_26 - var_17
        var_28 = var_27 + 1e-09
        var_29 = var_17 / var_28
        var_30 = paddle.nn.functional.loss.binary_cross_entropy_with_logits(
            var_0, var_29, reduction='none'
        )
        var_31 = var_30 * 1.0
        return var_31


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, -1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, -1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, -1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, -1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, -1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=True,
            ),
            InputSpec(
                shape=(-1, -1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=True,
            ),
            InputSpec(
                shape=(-1, -1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=True,
            ),
            InputSpec(
                shape=(-1, -1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=True,
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[1, 3, 96, 96, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 96, 96, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 96, 96, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 96, 96, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 96, 96, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 96, 96, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 96, 96, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 96, 96, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 96, 96, 1], dtype=paddle.float32),
        )
        self.net = LayerCase


if __name__ == '__main__':
    unittest.main()
