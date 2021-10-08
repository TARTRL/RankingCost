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


from argparse import Namespace


def merge_dict(dict1, dict2):
    res = {**dict1, **dict2}
    return res

HYPERPARAMS = {
    'ES_base': {
        'env': "ES_GR",
        'exp_name': 'ES_GR',
        'lr': 1e-3,
        'seed': 0,
        'action_type': 'VonNeumann_4',  # 'Moore_8'
        'width': 7,
        'height': 7,
        'train_num': 1000,
        'POPULATION_SIZE': 40,
        'solver': 'ES',
        'block_width': 0,
        'train_rank': True
    }

}
HYPERPARAMS['Ranking Cost'] = merge_dict(HYPERPARAMS['ES_base'],
                                 {
                                     'exp_name': 'Ranking Cost',
                                     'model': 'v0',
                                     'train_rank': True
                                 }
                                 )

def convert_params(params):
    return Namespace(**params)
