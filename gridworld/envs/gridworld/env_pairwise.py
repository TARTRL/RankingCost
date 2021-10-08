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

import gym
from gym.utils import seeding
import numpy as np
from gym import spaces
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .maze import trans1, trans2
from .maze import pos_type
from .maze_utils import generate_maze, parse_map

class MazeConfig:
    def __init__(self, start_num=1, width=32, height=32, complexity=0.2, density=0.2, maze_type='RandomMaze',
                 max_distance=None):
        self.maze_type = maze_type
        self.width = width
        self.height = height
        self.complexity = complexity
        self.density = density
        self.max_distance = max_distance
        self.start_num = start_num


class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, maze_config, max_step, pob_size=1, action_type='VonNeumann_4', obs_type='full', show_trace=False,
                 show=False, seed=None):
        self.seed(seed)
        self.maze_config = maze_config
        self.maze, self.start_poses, self.goal_poses = generate_maze(self.maze_config)

        self.maze_data = np.array(self.maze.get_maze())
        self.maze_size = self.maze_data.shape
        self.max_step = max_step
        self.step_now = 0
        self.show_trace = show_trace
        self.traces = []
        self.action_type = action_type
        self.obs_type = obs_type
        self.show = show

        self.pos_nows = None

        # Action space: 0: Up, 1: Down, 2: Left, 3: Right
        if self.action_type == 'VonNeumann_4':  # Von Neumann neighborhood
            self.num_actions = 4
        elif action_type == 'Moore_8':  # Moore neighborhood
            self.num_actions = 8
        else:
            raise TypeError('Action type must be either \'VonNeumann\' or \'Moore\'')

        self.action_space = spaces.Discrete(self.num_actions * self.maze_config.start_num)
        self.all_actions = list(range(self.action_space.n))

        self.pob_size = pob_size

        low_obs = 0  # Lowest integer in observation
        high_obs = 6  # Highest integer in observation

        if self.obs_type == 'full':
            self.observation_space = spaces.Box(low=low_obs,
                                                high=high_obs,
                                                shape=self.maze_size,
                                                dtype=np.float32)
        elif self.obs_type == 'partial':
            self.observation_space = spaces.Box(low=low_obs,
                                                high=high_obs,
                                                shape=(self.pob_size * 2 + 1, self.pob_size * 2 + 1),
                                                dtype=np.float32)
        else:
            raise TypeError('Observation type must be either \'full\' or \'partial\'')

        # Colormap: order of color is, free space, wall, agent, food, poison
        self.cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'gray'])
        self.bounds = [pos_type['blank'], pos_type['block'], pos_type['agent'], pos_type['goal'], 4, 5,
                       6]  # values for each color
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)

        self.ax_imgs = []  # For generating videos

    def load_map(self, map_path):

        self.start_poses, self.goal_poses = parse_map(map_path, self.maze)

    def set_state(self,state):
        full_map, pairs = state
        self.maze.maze = full_map
        self.start_poses = []
        self.goal_poses = []
        for pair in pairs:
            self.start_poses.append( [int(p) for p in pair[0]] )
            self.goal_poses.append( [int(p) for p in pair[1]])


    def reset(self, maze_config=None, new_map=True, new_start=True, new_goal=True, show=None, show_trace=None):
        if show is not None:
            self.show = show
        if show_trace is not None:
            self.show_trace = show_trace

        if maze_config is not None:
            new_map = False
            self.maze, self.start_poses, self.goal_poses = generate_maze(maze_config)

        if new_map:
            self.start_poses, self.goal_poses = self.maze.generate_maze()
        else:
            if new_goal and new_start:
                self.start_poses, self.goal_poses = self.maze.reset_starts_goals(self.maze_config.start_num)
            else:
                if new_goal:
                    self.goal_poses = self.maze.reset_goals(self.maze_config.start_num)
                if new_start:
                    self.start_poses = self.maze.reset_starts(self.maze_config.start_num)

        # print(self.start_pos)
        if hasattr(self, 'ani_obs'):
            self.ani_obs = []
            self.ani_obs_p = []

        self.maze_data = np.array(self.maze.get_maze())
        self.pos_nows = self.start_poses
        self.ax_imgs = []
        self.traces = self.start_poses
        self.step_now = 0
        return self._get_obs()

    def step(self, action):
        info = {}

        pre_pos = self.pos_now
        self.pos_now = self._next_pos(self.pos_now, action)

        self.traces.append(self.pos_now)

        if self._goal_test(self.pos_now):  # Goal check
            reward = +1
            done = True
        elif self.pos_now == pre_pos:  # Hit wall
            reward = -0.1
            done = False
        else:  # Moved, small negative reward to encourage shorest path
            reward = -0.01
            done = False

            # Additional info

        self.step_now += 1
        if self.step_now >= self.max_step:
            done = True
        return self._get_obs(), reward, done, info

    def seed(self, seed=None):
        if seed is None:
            return
        self.np_random, seed = seeding.np_random(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _get_obs(self):
        if self.obs_type == 'full':
            return self._get_full_obs()
        elif self.obs_type == 'partial':
            return self._get_partial_obs(self.pob_size)

    def _get_full_obs_v0(self):
        """Return a 2D array representation of maze."""
        obs = np.array(self.maze_data)
        # Set goal positions
        # for goal in self.goal_poses:
        #     print(goal[0],goal[1],pos_type['goal'])
        # exit()
        for goal in self.goal_poses:
            obs[goal[0]][goal[1]] = pos_type['goal']  # 3: goal

        # Set current position
        # Come after painting goal positions, avoid invisible within multi-goal regions
        obs[self.pos_now[0]][self.pos_now[1]] = pos_type['agent']  # 2: agent
        return obs

    def _get_full_obs(self):
        # return (maze_size,3) observation, first dim for maze, second dim for start, third dim for goal
        # return np.stack([self.maze_data, starts_map, goals_map])
        pairs = list(zip(self.start_poses, self.goal_poses))
        return self.maze_data, pairs

    def _get_partial_obs(self, size=1):
        """Get partial observable window according to Moore neighborhood"""
        # Get maze with indicated location of current position and goal positions
        maze = self._get_full_obs_v0()
        pos = np.array(self.pos_nows[0])

        under_offset = np.min(pos - size)
        over_offset = np.min(len(maze) - (pos + size + 1))
        offset = np.min([under_offset, over_offset])

        if offset < 0:  # Need padding
            maze = np.pad(maze, np.abs(offset), 'constant', constant_values=1)
            pos += np.abs(offset)

        return maze[pos[0] - size: pos[0] + size + 1, pos[1] - size: pos[1] + size + 1]

    def _goal_test(self, pos):
        """Return True if current state is a goal state."""
        if type(self.goal_poses[0]) == list:
            return list(pos) in self.goal_poses
        elif type(self.goal_poses[0]) == tuple:
            return tuple(pos) in self.goal_poses

    def _next_pos(self, pos, action):
        """Return the next state from a given state by taking a given action."""

        # Transition table to define movement for each action
        if self.action_type == 'VonNeumann_4':
            transitions = trans1
        elif self.action_type == 'Moore_8':
            transitions = trans2

        new_state = [pos[0] + transitions[action][0], pos[1] + transitions[action][1]]
        if self.maze_data[new_state[0]][new_state[1]] == 1:  # Hit wall, stay there
            return pos
        else:  # Valid move for 0, 2, 3, 4
            return new_state

    def render(self, mode='human', close=False):

        if close:
            plt.close()
            return

        obs = self._get_full_obs_v0()

        partial_obs = self._get_partial_obs(self.pob_size)

        # For rendering traces: Only for visualization, does not affect the observation data
        if self.show_trace:
            obs[tuple(list(zip(*self.traces[:-1])))] = 6

        for goal in self.goal_poses:
            obs[goal[0]][goal[1]] = 3  # 3: goal
        for pos_now in self.pos_nows:
            obs[pos_now[0]][pos_now[1]] = 2  # 2: agent

        if self.show:
            # Create Figure for rendering
            if not hasattr(self, 'fig'):  # initialize figure and plotting axes
                self.fig, (self.ax_full, self.ax_partial) = plt.subplots(nrows=1, ncols=2)
                self.ax_full.axis('off')
                self.ax_partial.axis('off')

            self.fig.show()

            # Only create the image the first time
            if not hasattr(self, 'ax_full_img'):
                self.ax_full_img = self.ax_full.imshow(obs, cmap=self.cmap, norm=self.norm, animated=True)
            if not hasattr(self, 'ax_partial_img'):
                self.ax_partial_img = self.ax_partial.imshow(partial_obs, cmap=self.cmap, norm=self.norm, animated=True)
            # Update the image data for efficient live video
            self.ax_full_img.set_data(obs)
            self.ax_partial_img.set_data(partial_obs)

            plt.draw()
            # Update the figure display immediately
            self.fig.canvas.draw()
            return self.fig

        else:
            if not hasattr(self, 'ani_obs'):
                self.ani_obs = [obs]
                self.ani_obs_p = [partial_obs]
            else:
                self.ani_obs.append(obs)
                self.ani_obs_p.append(partial_obs)

    def _get_video(self, interval=200, gif_path=None):
        if self.show:
            # TODO: Find a way to create animations without slowing down the live display
            print('Warning: Generating an Animation when live_display=True not yet supported.')

        if not hasattr(self, 'fig'):  # initialize figure and plotting axes
            self.fig, (self.ax_full, self.ax_partial) = plt.subplots(nrows=1, ncols=2)
            self.ax_full.axis('off')
            self.ax_partial.axis('off')
            self.fig.set_dpi(100)
        for obs, partial_obs in zip(self.ani_obs, self.ani_obs_p):
            # Create a new image each time to allow an animation to be created
            self.ax_full_img = self.ax_full.imshow(obs, cmap=self.cmap, norm=self.norm, animated=True)
            self.ax_partial_img = self.ax_partial.imshow(partial_obs, cmap=self.cmap, norm=self.norm, animated=True)
            # Put in AxesImage buffer for video generation
            self.ax_imgs.append([self.ax_full_img, self.ax_partial_img])  # List of axes to update figure frame

        anim = animation.ArtistAnimation(self.fig, self.ax_imgs, interval=interval)
        if gif_path is not None:
            anim.save(gif_path, writer='imagemagick', fps=10)

        return anim

    def get_maze(self):
        return self.maze.maze

    def get_goals(self):
        return self.goal_poses

    def get_starts(self):
        return self.start_poses
