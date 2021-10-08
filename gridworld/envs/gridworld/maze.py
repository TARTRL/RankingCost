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
import copy

import numpy as np
from numpy.random import randint
from PIL import Image


pos_type = {'blank': 0, 'block': 1, 'agent': 2, 'goal': 3}

pos_type_reverse = {}
for key in pos_type:
    pos_type_reverse[pos_type[key]] = key

type_color = {'blank': (255, 255, 255), 'block': (0, 0, 0), 'agent': (255, 255, 0), 'goal': (0, 255, 0),
              'trace': (100, 100, 100)}
type_color_reverse = {}
for key in type_color:
    type_color_reverse[type_color[key]] = key

# up,down,right,left
trans1 = {0: [-1, 0], 1: [1, 0], 2: [0, 1], 3: [0, -1]}  # VonNeumann_4
trans2 = {0: [-1, 0], 1: [1, 0], 2: [0, 1], 3: [0, -1],
          4: [-1, 1], 5: [-1, -1], 6: [1, 1], 7: [1, -1]}  # Moore_8
trans1_cost = {0: 1, 1: 1, 2: 1, 3: 1}
trans2_cost = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1.41421, 5: 1.41421, 6: 1.41421, 7: 1.41421}
default_pos_width = 20


def get_arrow(action, width, back_color):
    img = Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'icons/arrow.png'))
    img = img.resize((width, width))

    if action == 1:
        img = img.rotate(180)
    if action == 2:
        img = img.rotate(90)
    if action == 3:
        img = img.rotate(270)

    img = np.array(img)

    new_img = img[:,:,:3]

    place = np.where(img[:,:,3] == 0)
    
    new_img[place[0],place[1],:] = back_color
    return new_img


def solve_dis(maze,pos,max_dis,transitions):
    distance_map = np.zeros(np.shape(maze)) - 1
    state = copy.copy(maze)
    poses_now = [pos]
    distance_map[pos[0], pos[1]] = 0
    state[pos[0], pos[1]] = 2
    blank_space = []
    while len(poses_now) > 0:

        pos_now = poses_now[0]

        del poses_now[0]

        for key in transitions:
            next_pos = [pos_now[0] - transitions[key][0], pos_now[1] - transitions[key][1]]
            if state[next_pos[0], next_pos[1]] == 0:
                state[next_pos[0], next_pos[1]] = 2
                distance_map[next_pos[0], next_pos[1]] = distance_map[pos_now[0], pos_now[1]] + 1
                if distance_map[next_pos[0], next_pos[1]] <= max_dis:
                    poses_now.append(next_pos)
                    blank_space.append(next_pos)
    return blank_space

def blank_space_in_dis(maze,pos,dis):
    blank_space = solve_dis(maze,pos,dis,transitions=trans1)
    return blank_space

class MazeBase(object):

    def __init__(self):
        self.maze = None

    def generate_maze(self):
        return self._generate_maze()

    def parse_map(self, map_path):
        import json
        with open(map_path, 'r')  as f:
            map_info = json.load(f)
            self.maze = np.array([[int(it) for it in line_data] for line_data in map_info['map']])
            start_poses = []
            goal_poses = []
            pairs = map_info['pairs']
            for pair in pairs:
                start_poses.append(pair[0])
                goal_poses.append(pair[1])

        return start_poses, goal_poses

    def _generate_maze(self, maze=None, start=None, goals=None):
        raise NotImplementedError

    def reset_starts(self, start_num=1):
        self.start_poses = []
        for i in range(start_num):
            if self.max_distance and self.max_distance > 0 and hasattr(self, 'goal_poses') and len(
                    self.goal_poses) == start_num:

                blank_space = blank_space_in_dis(self.maze, self.goal_poses[i], self.max_distance)

                for goal in self.goal_poses:
                    blank_space.remove(goal)
                for start in self.start_poses:
                    if start in blank_space:
                        blank_space.remove(start)
            else:
                blank_space = np.where(self.maze == pos_type['blank'])
                blank_space = list(zip(blank_space[0], blank_space[1]))
                for start in self.start_poses:
                    if tuple(start) in blank_space:
                        blank_space.remove(tuple(start))
                if hasattr(self, 'goal_poses'):
                    for goal in self.goal_poses:
                        if tuple(goal) in blank_space:
                            blank_space.remove(tuple(goal))

            assert len(blank_space) >= 1

            start_pos_index = np.random.choice(len(blank_space), size=1, replace=False)[0]
            self.start_poses.append(list(blank_space[start_pos_index]))

        return self.start_poses

    def reset_goals(self, goal_num=1, ignore_start=False):
        self.goal_poses = []

        for i in range(goal_num):
            if not ignore_start and self.max_distance and self.max_distance > 0 and hasattr(self, 'start_poses'):
                blank_space = blank_space_in_dis(self.maze, self.start_poses[i], self.max_distance)
                for start in self.start_poses:
                    blank_space.remove(start)
                for goal in self.goal_poses:
                    if goal in blank_space:
                        blank_space.remove(goal)
            else:
                blank_space = np.where(self.maze == pos_type['blank'])
                blank_space = list(zip(blank_space[0], blank_space[1]))
                for goal in self.goal_poses:
                    if tuple(goal) in blank_space:
                        blank_space.remove(tuple(goal))
                if hasattr(self, 'start_poses'):
                    for start in self.start_poses:
                        if tuple(start) in blank_space:
                            blank_space.remove(tuple(start))

            assert len(blank_space) >= 1
            goal_pos_indexes = np.random.choice(len(blank_space), size=1, replace=False)
            goal_pos_index = goal_pos_indexes[0]
            goal_pos = list(blank_space[goal_pos_index])
            self.goal_poses.append(goal_pos)
        return self.goal_poses


    def reset_starts_goals(self, start_num):
        goal_poses = self.reset_goals(start_num, ignore_start=True)
        start_poses = self.reset_starts(start_num)
        return start_poses, goal_poses

    def get_maze(self):
        return self.maze

    def add_starts_and_goals(self):
        new_maze = np.array(self.maze)
        if hasattr(self, 'start_poses'):
            for start in self.start_poses:
                new_maze[start[0], start[1]] = pos_type['agent']
        if hasattr(self, 'goal_poses'):
            for goal in self.goal_poses:
                new_maze[goal[0], goal[1]] = pos_type['goal']
        return new_maze

    def clear_starts_and_goals(self):
        agent_poses = np.where(self.maze == pos_type['agent'])
        agent_poses = list(zip(agent_poses[0], agent_poses[1]))

        if len(agent_poses) > 0:
            self.start_poses = agent_poses
            for agent_pos in agent_poses:
                self.maze[agent_pos[0], agent_pos[1]] = pos_type['blank']

        goal_poses = np.where(self.maze == pos_type['goal'])
        goal_poses = list(zip(goal_poses[0], goal_poses[1]))

        if len(goal_poses) > 0:
            self.goal_poses = goal_poses

            for goal in goal_poses:
                self.maze[goal[0], goal[1]] = pos_type['blank']

    def save(self,save_path):
        new_maze = self.add_starts_and_goals()
        np.save(save_path, new_maze)


    def load(self,load_path):
        self.maze = np.load(load_path)
        self.clear_starts_and_goals()

    def load_img(self,load_path, pos_width = default_pos_width):
        if not load_path.split('.')[-1] == 'png':
            print('error! image type must be png!')
            exit()
        img = Image.open(load_path)
        img.load()
        data = np.asarray(img, dtype="int32")
        img_shape = np.shape(data)

        shape = (int(img_shape[0]/pos_width),int(img_shape[1]/pos_width))

        self.maze = np.zeros(shape)

        for  r in range(shape[0]):
            for c in range(shape[1]):
                self.maze[r,c] = pos_type[type_color_reverse[tuple(data[r*pos_width,c*pos_width,:])]]

        self.clear_starts_and_goals()

class RandomMaze(MazeBase):

    def __init__(self, width=81, height=51, complexity=0.75, density=0.75, max_distance=None, start_num=1):
        super().__init__()

        self.width = width
        self.height = height
        self.complexity = complexity
        self.density = density


        self.shape = (self.height, self.width)
        # Adjust complexity and density relative to maze size
        self.complexity = int(self.complexity * (5 * (self.shape[0] + self.shape[1])))
        self.density = int(self.density * ((self.shape[0] // 2) * (self.shape[1] // 2)))
        self.max_distance = max_distance
        self.start_num = start_num

    def _generate_maze(self):

        """
        Code from https://en.wikipedia.org/wiki/Maze_generation_algorithm
        """

        # Build actual maze

        Z = np.zeros(self.shape, dtype=bool)
        # Fill borders
        Z[0, :] = Z[-1, :] = pos_type['block']
        Z[:, 0] = Z[:, -1] = pos_type['block']
        # Make aisles
        for i in range(self.density):


            x, y = min(randint(0, (self.shape[1]-1) // 2 + 1) * 2, self.shape[1] - 2), min(
                randint(0, (self.shape[0]-1) // 2 + 1) * 2, self.shape[0] - 2)
            Z[y, x] = pos_type['block']
            for j in range(self.complexity):
                neighbours = []
                if x > 1:             neighbours.append((y, x - 2))
                if x < self.shape[1] - 2:  neighbours.append((y, x + 2))
                if y > 1:             neighbours.append((y - 2, x))
                if y < self.shape[0] - 2:  neighbours.append((y + 2, x))
                if len(neighbours):
                    y_, x_ = neighbours[randint(0, len(neighbours))]
                    if Z[y_, x_] == pos_type['blank']:
                        Z[y_, x_] = pos_type['block']
                        Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_
        self.maze = Z.astype(int)
        return self.reset_starts_goals(start_num=self.start_num)

