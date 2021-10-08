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
from PIL import Image, ImageDraw, ImageFont

from .maze import RandomMaze
from .maze import pos_type, pos_type_reverse, type_color, get_arrow

default_color = (0, 0, 0)
FontData = os.path.join(os.path.dirname(os.path.realpath(__file__)), "OpenSans-Regular.ttf")

font = ImageFont.truetype(FontData, size=12)


def get_pa1(width, background, color):
    a1 = np.zeros([width, width, 3]) + background
    for i in range(width):
        a1[i, width - i - 1:] = color
    return a1


def get_pa2(width, background, color):
    a2 = np.zeros([width, width, 3]) + background
    for i in range(width):
        a2[width - i - 1, :i + 1] = color
    return a2


def get_pa3(width, background, color):
    a3 = np.zeros([width, width, 3]) + background
    for i in range(width):
        a3[i, :i + 1] = color
    return a3


def get_pa4(width, background, color):
    a4 = np.zeros([width, width, 3]) + background
    for i in range(width):
        a4[i, i:] = color
    return a4


def generate_maze(maze_config):
    maze = None

    if maze_config.maze_type == 'RandomMaze':
        maze = RandomMaze(width=maze_config.width, height=maze_config.height
                          , complexity=maze_config.complexity, density=maze_config.density,
                          max_distance=maze_config.max_distance, start_num=maze_config.start_num)
    assert maze is not None

    start_poses, goal_poses = maze.generate_maze()
    return maze, start_poses, goal_poses


def parse_map(map_path, maze):
    start_poses, goal_poses = maze.parse_map(map_path)
    return start_poses, goal_poses


def save_json(state, save_path):
    import json
    import jsbeautifier

    full_map, pairs = state
    full_map = full_map.astype(np.int16).tolist()

    save_dict = {}
    save_dict['map'] = full_map
    save_dict['pairs'] = []
    for i in range(len(pairs)):
        save_dict['pairs'].append([[int(ii) for ii in pairs[i][0]], [int(ii) for ii in pairs[i][1]]])

    with open(save_path, 'w') as json_file:
        json.dump(save_dict, json_file)

    res = jsbeautifier.beautify_file(save_path)
    with open(save_path, 'w') as json_file:
        json_file.write(res)


def saveImage(state, save_path, action=None, pos_width=20):
    if not save_path.split('.')[-1] == 'png':
        print('error! image type must be png!')
        exit()
    full_map, pairs = state

    shape = np.shape(full_map)

    img_shape = (tuple([_ * pos_width for _ in shape]) + (3,))

    img = np.zeros(img_shape, dtype=np.uint8)

    full_map = state[0]


    for r in range(shape[0]):
        for c in range(shape[1]):
            img[r * pos_width:(r + 1) * pos_width, c * pos_width:(c + 1) * pos_width, :] = type_color[
                pos_type_reverse[full_map[r, c]]]

    if action is not None:
        agent_poses = np.where(full_map == pos_type['agent'])

        agent_poses = [int(agent_poses[0]), int(agent_poses[1])]

        for agent_pos in agent_poses:
            img[agent_pos[0] * pos_width:(agent_pos[0] + 1) * pos_width,
            agent_pos[1] * pos_width:(agent_pos[1] + 1) * pos_width, :] = get_arrow(action, pos_width,
                                                                                    type_color['agent'])


    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    w_shift = 3
    point_correct = -1
    for i, pair in enumerate(pairs):
        p_start = pair[0]
        p_end = pair[1]

        draw.ellipse((p_start[1] * pos_width, p_start[0] * pos_width, (p_start[1] + 1) * pos_width + point_correct,
                      (p_start[0] + 1) * pos_width + point_correct), fill=type_color['agent'])
        draw.ellipse((p_end[1] * pos_width, p_end[0] * pos_width, (p_end[1] + 1) * pos_width + point_correct,
                      (p_end[0] + 1) * pos_width + point_correct), fill=type_color['goal'])

        draw.text((p_start[1] * pos_width + w_shift, p_start[0] * pos_width), 'S{}'.format(i + 1), fill=default_color,
                  font=font)
        draw.text((p_end[1] * pos_width + w_shift, p_end[0] * pos_width), 'E{}'.format(i + 1), fill=default_color,
                  font=font)
    img.save(save_path)


def save_solution_json(file, solver,time_usage=None):
    import json
    re_dict = {}

    re_dict['cost'] = float(solver.cost_now)
    re_dict['traces'] = solver.traces_now

    if time_usage:
        re_dict['time_usage'] = time_usage

    with open(file, 'w') as f:
        json.dump(re_dict, f)


def saveSolution(state, save_path, solver=None, pos_width=20, save_json=True,time_usage=None):
    if not save_path.split('.')[-1] == 'png':
        print('error! image type must be png!')
        exit()
    json_file = save_path[:-3] + 'json'

    full_map, pairs = state

    shape = np.shape(full_map)

    img_shape = (tuple([_ * pos_width for _ in shape]) + (3,))

    img = np.zeros(img_shape, dtype=np.uint8)

    full_map = state[0]

    for r in range(shape[0]):
        for c in range(shape[1]):
            img[r * pos_width:(r + 1) * pos_width, c * pos_width:(c + 1) * pos_width, :] = type_color[
                pos_type_reverse[full_map[r, c]]]

    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    if solver is not None and solver.traces_now is not None:
        if save_json:
            save_solution_json(json_file, solver,time_usage)
        for trace in solver.traces_now:
            color_now = (random.randint(10, 125), random.randint(10, 125), random.randint(10, 125))

            draw.line([(int((p[1] + 0.5) * pos_width), int((p[0] + 0.5) * pos_width)) for p in trace],
                      fill=color_now, width=5)
            cc = 2
            for p in trace:
                draw.ellipse(
                    (int((p[1] + 0.5) * pos_width) - cc,
                     int((p[0] + 0.5) * pos_width) - cc,
                     int((p[1] + 0.5) * pos_width) + cc,
                     int((p[0] + 0.5) * pos_width) + cc), fill=color_now)

    else:
        if save_json:
            import json
            re_dict = {}
            re_dict['cost'] = None
            re_dict['traces'] = None
            re_dict['solution'] = None
            if time_usage:
                re_dict['time_usage'] = time_usage
            with open(json_file, 'w') as f:
                json.dump(re_dict, f)

    w_shift = 3
    point_correct = -1
    for i, pair in enumerate(pairs):
        p_start = pair[0]
        p_end = pair[1]

        draw.ellipse((p_start[1] * pos_width, p_start[0] * pos_width, (p_start[1] + 1) * pos_width + point_correct,
                      (p_start[0] + 1) * pos_width + point_correct), fill=type_color['agent'])
        draw.ellipse((p_end[1] * pos_width, p_end[0] * pos_width, (p_end[1] + 1) * pos_width + point_correct,
                      (p_end[0] + 1) * pos_width + point_correct), fill=type_color['goal'])

        draw.text((p_start[1] * pos_width + w_shift, p_start[0] * pos_width), 'S{}'.format(i + 1), fill=default_color,
                  font=font)
        draw.text((p_end[1] * pos_width + w_shift, p_end[0] * pos_width), 'E{}'.format(i + 1), fill=default_color,
                  font=font)

    img.save(save_path)