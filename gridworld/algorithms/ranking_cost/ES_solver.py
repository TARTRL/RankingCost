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

import pickle
import copy

import numpy as np


from gridworld.envs.gridworld.multiGoalSolver import SelectSolver
from gridworld.algorithms.ranking_cost.ES_model import ESModel
from gridworld.algorithms.ranking_cost.ES_engine import EvolutionStrategy

best_step = 50


class ESSolver(SelectSolver):
    AGENT_HISTORY_LENGTH = 1
    REWARD_SCALE = 20
    SIGMA = 0.1
    LEARNING_RATE = 0.01

    def __init__(self, train_num, POPULATION_SIZE=20, print_step=1, action_type='VonNeumann_4', block_width=0,
                 train_rank=True, model_save_path=None, logger=None, saver=None):
        super().__init__(action_type, block_width)
        self.train_num = train_num
        self.POPULATION_SIZE = POPULATION_SIZE
        self.print_step = print_step
        self.train_rank = train_rank
        self.model_save_path = model_save_path if model_save_path else './model.pkl'

        self.logger = logger
        self.saver = saver
        self.es = EvolutionStrategy(self.get_reward, self.log_function, self.stop_function,
                                    self.saver,
                                    self.POPULATION_SIZE, self.SIGMA,
                                    self.LEARNING_RATE)

    def _solve(self, maze_data, starts, goals):
        assert len(starts) == len(goals)
        sg_pairs = list(zip(starts, goals))

        start_num = len(sg_pairs)
        self.mini_cost = 1e6
        self.max_cost = np.prod(maze_data.shape)

        self.best_solution = None
        self.best_traces = None
        self.best_reward = -1e6
        self.best_reward_step = 0

        self.model = ESModel(pair_num=start_num, maze_shape=maze_data.shape)
        self.es.reset()
        self.es.set_weights(self.model.get_weights())

        self.maze_data = maze_data
        self.sg_pairs = sg_pairs

        self.es.run(iterations=self.train_num)

        if self.saver:
            print('save final model!')
            self.saver.save(self.es.get_weights())

    def close(self):
        self.es.close()

    def log_function(self, iteration, weights, reward, rewards, time_duration):
        if self.logger:
            self.logger.update_data({'reward': reward,
                                     'rewards': rewards,
                                     'time': time_duration,
                                     'step': iteration})
            self.logger.display_info()
            self.logger.plot_figure()
            self.logger.save_solution(self)

    def stop_function(self, iteration, weights, reward, rewards, time_duration):
        if self.best_reward > reward:
            self.best_reward = reward
            self.best_reward_step = iteration
            return False
        if self.best_reward <= reward and (iteration - self.best_reward_step) >= best_step:
            return True

        return False

    def get_reward(self, weights):
        cost_now, traces_now, solution_rank = self.get_solution(weights)
        if cost_now is None or cost_now >= self.max_cost:
            cost_now = self.max_cost
        cost_now = cost_now / self.max_cost
        reward_now = -cost_now
        return reward_now * self.REWARD_SCALE

    def get_solution(self, weights):
        self.model.set_weights(weights)
        solution_rank, maze_weights = self.get_maze_weights()

        cost_now, traces_now = self.planOnSolution(solution_rank, maze_weights, self.maze_data, self.sg_pairs)
        self.cost_now = cost_now
        self.traces_now = traces_now
        self.solution_now = solution_rank
        return cost_now, traces_now, solution_rank

    def get_maze_weights(self):
        prediction = self.model.get_rank_prob()
        if self.train_rank:
            solution_rank = np.argsort(prediction)[::-1]
        else:
            solution_rank = range(len(prediction))

        maze_now = copy.copy(self.model.get_maze_data())

        maze_now = maze_now[solution_rank]
        return solution_rank, maze_now

    def planOnSolution(self, rank_now, maze_weights, maze_data, sg_pairs):
        rank_now = list(rank_now)
        cost_all = 0
        new_maze_data = copy.copy(maze_data)
        new_sg_pairs = []
        for i in rank_now:
            new_sg_pairs.append(sg_pairs[i])

        traces = []
        for i, sg_pair in enumerate(new_sg_pairs):

            block_poses = []

            if i < len(new_sg_pairs) - 1:
                for b in new_sg_pairs[i + 1:]:
                    for bb in b:
                        block_poses.append(bb)
                        if self.block_width > 0:
                            for key in self.transitions:
                                bb_neighbour = [bb[0] + self.transitions[key][0], bb[1] + self.transitions[key][1]]

                                block_poses.append(bb_neighbour)
            if i == len(new_sg_pairs) - 1:
                extra_cost = None
            else:
                extra_cost = np.sum(maze_weights[i + 1:], axis=0)

            cost, trace_now = self.planOnOneStart(new_maze_data, sg_pair, block_poses, extra_cost=extra_cost)

            if cost is None:
                return None, None
            traces.append(trace_now)

            cost_all += cost

        return cost_all, traces

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            self.model.set_weights(pickle.load(fp))
        self.es.weights = self.model.get_weights()
