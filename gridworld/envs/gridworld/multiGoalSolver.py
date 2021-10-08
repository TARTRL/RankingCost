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

import random
import copy

import math

import numpy as np
from .maze import trans1, trans2, trans1_cost, trans2_cost

from operator import attrgetter


class Node:
    def __init__(self, position, g_value, h_value, pre_node=None, g_value_real=None):
        self.position = position
        self.g_value = g_value
        if g_value_real is None:
            self.g_value_real = g_value
        else:
            self.g_value_real = g_value_real
        self.h_value = h_value
        self.pre_node = pre_node
        self.f_value = g_value + h_value

    def __eq__(self, other):
        return self.position == other.position


class SolverBase(object):
    def __init__(self, action_type='VonNeumann_4'):
        self.action_type = action_type
        if action_type == 'VonNeumann_4':
            self.transitions = trans1
            self.transitions_cost = trans1_cost
        elif action_type == 'Moore_8':
            self.transitions = trans2
            self.transitions_cost = trans2_cost

    def _solve(self, maze_data, starts, goals):
        pass

    def solve(self, maze_data, starts, goals):
        self._solve(maze_data, starts, goals)

    def action(self, state):
        self.state = state
        return self._action()

    def _action(self):
        pass

    def reset(self, env, state=None):
        self.state = state
        self.solve(env.get_maze(), env.get_starts(), env.get_goals())

    def close(self):
        pass

    def load_solution(self,solution_path):
        import json
        with open(solution_path, 'r')  as f:
            solution = json.load(f)
            self.cost_now = solution['cost']
            self.traces_now = solution['traces']
            if self.traces_now == None:
                return False
            else:
                return True


class SelectSolver(SolverBase):
    def __init__(self, action_type='VonNeumann_4', block_width=0, sample_num=-1):
        super().__init__(action_type)
        self.block_width = block_width
        self.sample_num = sample_num


    def _solve(self, maze_data, starts, goals):

        assert len(starts) == len(goals)
        sg_pairs = list(zip(starts, goals))

        start_num = len(sg_pairs)
        self.mini_cost = 1e6
        self.best_solution = None
        self.best_traces = None

        check_list = []
        a = list(range(start_num))

        self.max_sample_num = math.factorial(start_num)

        while len(check_list)<self.sample_num and len(check_list)<self.max_sample_num:
            random.shuffle(a)
            if not a in check_list:
                check_list.append(copy.copy(a))

        solution_index = 0



        for solution_now in check_list:
            solution_index += 1
            cost_now, traces_now = self.planOnSolution(solution_now, maze_data, sg_pairs)

            if cost_now is None:
                continue

            if cost_now < self.mini_cost:
                self.best_solution = solution_now
                self.mini_cost = cost_now
                self.best_traces = traces_now


        self.traces_now = self.best_traces
        self.solution_now = self.best_solution
        self.cost_now = self.mini_cost

    def planOnSolution(self, solution_now, maze_data, sg_pairs):
        solution_now = list(solution_now)
        cost_all = 0
        new_maze_data = copy.copy(maze_data)
        new_sg_pairs = []
        for i in solution_now:
            new_sg_pairs.append(sg_pairs[i])

        traces = []
        for i, sg_pair in enumerate(new_sg_pairs):

            block_poses = []

            if i < len(new_sg_pairs) - 1:
                for b in new_sg_pairs[i + 1:]:
                    for bb in b:
                        block_poses.append(bb)
                        if self.block_width>0:
                            for key in self.transitions:
                                bb_neighbour = [bb[0] + self.transitions[key][0], bb[1] + self.transitions[key][1]]

                                block_poses.append(bb_neighbour)

            cost, trace_now = self.planOnOneStart(new_maze_data, sg_pair, block_poses)


            if cost is None:
                return None, None
            traces.append(trace_now)

            cost_all += cost

        return cost_all, traces

    def planOnOneStart(self, new_maze_data, sg_pair, block_poses, method='A*',extra_cost=None):
        if method == 'A*':
            return self.planOnOneStart_Astar(new_maze_data, sg_pair, block_poses,extra_cost)
        if method == 'BFS':
            return self.planOnOneStart_BFS(new_maze_data, sg_pair, block_poses,extra_cost)
        assert False

    def planOnOneStart_Astar(self, new_maze_data, sg_pair, block_poses,extra_cost=None):

        maze_now = copy.copy(new_maze_data)
        maze_shape = maze_now.shape

        for block_pos in block_poses:

            maze_now[block_pos[0], block_pos[1]] = 1

        def get_h_value(point_now):
            return math.sqrt((point_now[0] - sg_pair[1][0]) ** 2 + (point_now[1] - sg_pair[1][1]) ** 2)
        def get_extra_value(point_now):
            if extra_cost is None:
                return 0
            else:
                return extra_cost[point_now[0],point_now[1]]

        open_list = [
            Node([sg_pair[0][0], sg_pair[0][1]], g_value=0, h_value=get_h_value([sg_pair[0][0], sg_pair[0][1]]))]


        while len(open_list) > 0:

            node_now = min(open_list, key=attrgetter('f_value'))
            node_now_index = open_list.index(node_now)
            pos_now = node_now.position
            maze_now[node_now.position[0], node_now.position[1]] = 2
            open_list.pop(node_now_index)


            for key in self.transitions:
                next_pos = [pos_now[0] + self.transitions[key][0], pos_now[1] + self.transitions[key][1]]
                if next_pos[0]>=0 and next_pos[1]>=0 and next_pos[0] <maze_shape[0] and next_pos[1] < maze_shape[1] and maze_now[next_pos[0], next_pos[1]] == 0:
                    next_node = Node(next_pos,
                                     g_value=node_now.g_value + self.transitions_cost[key] + get_extra_value(next_pos),
                                     g_value_real=node_now.g_value_real + self.transitions_cost[key],
                                     h_value=get_h_value(next_pos),
                                     pre_node=node_now)

                    if next_node in open_list:
                        next_index = open_list.index(next_node)
                        next_pre = open_list[next_index]
                        if next_pre.g_value > next_node.g_value:
                            open_list[next_index] = next_node
                    else:
                        open_list.append(next_node)

                    if next_pos == sg_pair[1]:
                        trace_now = [sg_pair[1]]
                        pre_node = next_node.pre_node
                        while pre_node:
                            new_maze_data[trace_now[-1][0], trace_now[-1][1]] = 1

                            if self.block_width > 0:
                                for key in self.transitions:
                                    neighbour = [trace_now[-1][0] + self.transitions[key][0],
                                                 trace_now[-1][1] + self.transitions[key][1]]
                                    if neighbour[0] >= 0 and neighbour[1] >= 0 and neighbour[0] < maze_shape[0] and neighbour[1] < maze_shape[1]:
                                        new_maze_data[neighbour[0], neighbour[1]] = 1


                            trace_now.append(pre_node.position)
                            pre_node = pre_node.pre_node

                        new_maze_data[trace_now[-1][0], trace_now[-1][1]] = 1
                        if self.block_width > 0:
                            for key in self.transitions:
                                neighbour = [trace_now[-1][0] + self.transitions[key][0],
                                             trace_now[-1][1] + self.transitions[key][1]]
                                if neighbour[0] >= 0 and neighbour[1] >= 0 and neighbour[0] < maze_shape[0] and neighbour[1] < maze_shape[1]:
                                    new_maze_data[neighbour[0], neighbour[1]] = 1

                        return next_node.g_value_real, trace_now[::-1]

        return None, None

    def planOnOneStart_BFS(self, new_maze_data, sg_pair, block_poses,extra_cost=None):


        maze_now = copy.copy(new_maze_data)
        for block_pos in block_poses:
            maze_now[block_pos[0], block_pos[1]] = 1

        action_map = np.zeros(shape=np.shape(maze_now), dtype=np.int32)
        distance_map = np.zeros(np.shape(maze_now)) - 1

        poses_now = [sg_pair[1]]

        distance_map[sg_pair[1][0], sg_pair[1][1]] = 0
        maze_now[sg_pair[1][0], sg_pair[1][1]] = 2
        maze_shape = maze_now.shape
        while len(poses_now) > 0:
            pos_now = poses_now[0]
            del poses_now[0]
            for key in self.transitions:
                next_pos = [pos_now[0] - self.transitions[key][0], pos_now[1] - self.transitions[key][1]]
                if next_pos[0]>=0 and next_pos[1]>=0 and next_pos[0] <maze_shape[0] and next_pos[1] < maze_shape[1] and maze_now[next_pos[0], next_pos[1]] == 0:
                    maze_now[next_pos[0], next_pos[1]] = 2
                    distance_map[next_pos[0], next_pos[1]] = distance_map[pos_now[0], pos_now[1]] + \
                                                             self.transitions_cost[key]
                    action_map[next_pos[0], next_pos[1]] = key
                    poses_now.append(next_pos)
                    if next_pos == sg_pair[0]:
                        trace_now = [sg_pair[0]]

                        while trace_now[-1] != sg_pair[1]:
                            new_maze_data[trace_now[-1][0], trace_now[-1][1]] = 1
                            if self.block_width > 0:
                                for key in self.transitions:
                                    neighbour = [trace_now[-1][0] + self.transitions[key][0],
                                                 trace_now[-1][1] + self.transitions[key][1]]

                                    new_maze_data[neighbour[0], neighbour[1]] = 1

                            action_now = action_map[trace_now[-1][0], trace_now[-1][1]]

                            trace_now.append([trace_now[-1][0] + self.transitions[action_now][0],
                                              trace_now[-1][1] + self.transitions[action_now][1]])

                        new_maze_data[trace_now[-1][0], trace_now[-1][1]] = 1
                        if self.block_width > 0:
                            for key in self.transitions:
                                neighbour = [trace_now[-1][0] + self.transitions[key][0],
                                             trace_now[-1][1] + self.transitions[key][1]]
                                new_maze_data[neighbour[0], neighbour[1]] = 1

                        return distance_map[next_pos[0], next_pos[1]], trace_now

        return None, None

    def _action(self):
        agent_pos = np.where(self.state[1] == 1)
        self.agent_pos = list(zip(agent_pos[0], agent_pos[1]))[0]
        return self.action_map[self.agent_pos[0], self.agent_pos[1]]
