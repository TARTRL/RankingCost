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

import copy
import numpy as np

from .maze import trans1, trans2


class SolverBase(object):
    def __init__(self, action_type='VonNeumann_4'):
        if action_type == 'VonNeumann_4':
            self.transitions = trans1
        elif action_type == 'Moore_4':
            self.transitions = trans2

    def _solve(self, maze_data, goals):
        pass

    def solve(self, maze_data, goals):
        self._solve(maze_data, goals)

    def action(self, state):
        self.state = state
        return self._action()

    def _action(self):
        print('ininin22s')
        pass

    def reset(self,env):
        self.solve(env.get_maze(),env.get_goals())

class OneGoalSolver(SolverBase):

    def __init__(self, action_type='VonNeumann_4'):
        super().__init__(action_type)

    def _solve(self, maze_data, goals):

        self.action_map = np.zeros(shape=np.shape(maze_data), dtype=np.int32)
        self.distance_map = np.zeros(np.shape(maze_data)) - 1

        state = copy.copy(maze_data)

        poses_now = [goals[0]]

        self.distance_map[goals[0][0], goals[0][1]] = 0
        state[goals[0][0], goals[0][1]] = 2

        while len(poses_now) > 0:

            pos_now = poses_now[0]

            del poses_now[0]

            for key in self.transitions:

                next_pos = [pos_now[0] - self.transitions[key][0], pos_now[1] - self.transitions[key][1]]
                if state[next_pos[0], next_pos[1]] == 0:
                    state[next_pos[0], next_pos[1]] = 2
                    self.distance_map[next_pos[0], next_pos[1]] = self.distance_map[pos_now[0], pos_now[1]] + 1
                    self.action_map[next_pos[0], next_pos[1]] = key
                    poses_now.append(next_pos)

    def _action(self):
        agent_pos = np.where(self.state[1] == 1)
        self.agent_pos = list(zip(agent_pos[0], agent_pos[1]))[0]
        return self.action_map[self.agent_pos[0], self.agent_pos[1]]
