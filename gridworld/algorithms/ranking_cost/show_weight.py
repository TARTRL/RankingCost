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

import pickle
import numpy as np
import matplotlib.pyplot as plt

def save_cost_maps(save_dir, name, weight):
    heat_maps = weight[1]

    for h_i, heat_map in enumerate(heat_maps):
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))


        im = ax.imshow(heat_map)
        cbar = ax.figure.colorbar(im, ax=ax, cmap="YlGn")
        cbar.ax.set_ylabel('color bar', rotation=-90, va="bottom")
        for edge, spine in ax.spines.items():
            spine.set_visible(False)
        ax.set_xticks(np.arange(heat_map.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(heat_map.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(heat_map.shape[0]):
            for j in range(heat_map.shape[1]):

                text = ax.text(j, i, '{:.1f}'.format(heat_map[i, j]),
                               ha="center", va="center", color="w", fontsize=60)

        fig.tight_layout()
        save_path = os.path.join(save_dir, '{}_{}.png'.format(name, h_i))
        print('save cost map to:',save_path)
        plt.savefig(save_path)


def save_cost_maps_v0(save_dir, name, weight):
    heat_maps = np.maximum(weight[1], 0) * 10

    for h_i, heat_map in enumerate(heat_maps):
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        im = ax.imshow(heat_map)
        cbar = ax.figure.colorbar(im, ax=ax, cmap="YlGn")
        cbar.ax.set_ylabel('color bar', rotation=-90, va="bottom")
        for edge, spine in ax.spines.items():
            spine.set_visible(False)
        ax.set_xticks(np.arange(heat_map.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(heat_map.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(heat_map.shape[0]):
            for j in range(heat_map.shape[1]):

                text = ax.text(j, i, int(heat_map[i, j]),
                               ha="center", va="center", color="w", fontsize=60)

        fig.tight_layout()
        save_path = os.path.join(save_dir, '{}_{}.png'.format(name, h_i))
        plt.savefig(save_path)

def load_weight():
    weight_file = './images/solver_map02_0_True_VonNeumann.pkl'
    with open(weight_file, 'rb') as f:
        weights = pickle.load(f)

        heat_maps = np.maximum(weights[1], 0) * 10
        map_index = 0
        for heat_map in heat_maps:
            fig, ax = plt.subplots(1, 1, figsize=(15, 15))

            im = ax.imshow(heat_map)
            cbar = ax.figure.colorbar(im, ax=ax, cmap="YlGn")
            cbar.ax.set_ylabel('color bar', rotation=-90, va="bottom")
            for edge, spine in ax.spines.items():
                spine.set_visible(False)
            ax.set_xticks(np.arange(heat_map.shape[1] + 1) - .5, minor=True)
            ax.set_yticks(np.arange(heat_map.shape[0] + 1) - .5, minor=True)
            ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)

            for i in range(heat_map.shape[0]):
                for j in range(heat_map.shape[1]):
                    # print(i,j)
                    text = ax.text(j, i, int(heat_map[i, j]),
                                   ha="center", va="center", color="w")
            ax.set_title("heat map")
            fig.tight_layout()
            plt.savefig('./images/heat_maps/{}.png'.format(map_index))
            map_index += 1


if __name__ == '__main__':
    load_weight()
