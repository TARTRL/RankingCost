#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 The TARTRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


class ESModel(object):
    def __init__(self, pair_num, maze_shape):
        rank_weights = np.zeros(shape=pair_num)
        maze_weights = np.zeros(shape=(pair_num, *maze_shape))
        self.weights = [rank_weights, maze_weights]

    def get_rank_prob(self):
        out = self.weights[0]
        return out

    def get_maze_data(self):
        out = np.maximum(self.weights[1], 0) * 10.

        return out

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights[0] = np.clip(weights[0], a_min=-10., a_max=10.)
        self.weights[1] = weights[1]
