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

import os
import random

import numpy as np

from gridworld.envs.gridworld.env_pairwise import MazeEnv as Env
from gridworld.envs.gridworld.env_pairwise import MazeConfig
from gridworld.envs.gridworld.maze_utils import saveSolution
from gridworld.parameters import convert_params, HYPERPARAMS
from gridworld.utils import get_filename
from gridworld.algorithms.ranking_cost.show_weight import save_cost_maps
from gridworld.algorithms.ranking_cost.ES_solver import ESSolver as Solver

def set_seed(params, envs):
    seed = params.seed

    np.random.seed(seed)
    random.seed(seed)

    for env in envs:
        env.seed(seed)
        env.action_space.seed(seed)


def run(params, loaded_map):
    log_save_dir = './'
    width = params.width
    height = params.height
    action_type = params.action_type
    maze_config = MazeConfig(start_num=4, width=width, height=height, complexity=0.05, density=0.05, max_distance=None)
    env = Env(maze_config, max_step=(width + height) * 2, show=False, seed=8, action_type=action_type)

    set_seed(params, [env])

    solver = Solver(POPULATION_SIZE=params.POPULATION_SIZE, action_type=action_type, train_num=params.train_num,
                    block_width=params.block_width)
    json_file = loaded_map
    file_name = get_filename(json_file)
    env.load_map(json_file)
    image_path = os.path.join(log_save_dir, '{}_solution.png'.format(file_name))
    state = env.reset(show=False, show_trace=False, new_map=False, new_start=False, new_goal=False)
    solver.reset(env, state)
    saveSolution(state, image_path, solver)
    save_cost_maps(save_dir=log_save_dir, name=get_filename(json_file), weight=solver.model.get_weights())
    solver.close()

def run_algo(exp_name, loaded_map):
    hyper_params = HYPERPARAMS[exp_name]
    params = convert_params(hyper_params)
    run(params, loaded_map)

if __name__ == '__main__':
    import sys
    exp_name = 'Ranking Cost'
    loaded_map = sys.argv[1]
    run_algo(exp_name, loaded_map)
