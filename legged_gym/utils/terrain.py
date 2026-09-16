# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


def _box_mesh(bounds_min, bounds_max):
    x0, y0, z0 = bounds_min
    x1, y1, z1 = bounds_max
    vertices = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)
    triangles = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=np.uint32)
    return vertices, triangles


def _append_box(vertices, triangles, bounds_min, bounds_max):
    box_vertices, box_triangles = _box_mesh(bounds_min, bounds_max)
    vertex_offset = sum(array.shape[0] for array in vertices)
    vertices.append(box_vertices)
    triangles.append(box_triangles + vertex_offset)


def _append_square_annulus(vertices, triangles, center, outer_half, inner_half, bottom, top):
    cx, cy = center
    _append_box(vertices, triangles, (cx-outer_half, cy-outer_half, bottom),
                (cx-inner_half, cy+outer_half, top))
    _append_box(vertices, triangles, (cx+inner_half, cy-outer_half, bottom),
                (cx+outer_half, cy+outer_half, top))
    _append_box(vertices, triangles, (cx-inner_half, cy-outer_half, bottom),
                (cx+inner_half, cy-inner_half, top))
    _append_box(vertices, triangles, (cx-inner_half, cy+inner_half, bottom),
                (cx+inner_half, cy+outer_half, top))


def _append_square_nosing(vertices, triangles, center, boundary_half, top,
                          depth, height, extends_outward):
    cx, cy = center
    overlap = 0.001
    if extends_outward:
        inner = boundary_half - overlap
        outer = boundary_half + depth
        span = outer
    else:
        inner = boundary_half - depth
        outer = boundary_half + overlap
        span = boundary_half + overlap
    bottom = top - height
    _append_box(vertices, triangles, (cx-outer, cy-span, bottom),
                (cx-inner, cy+span, top))
    _append_box(vertices, triangles, (cx+inner, cy-span, bottom),
                (cx+outer, cy+span, top))
    _append_box(vertices, triangles, (cx-span, cy-outer, bottom),
                (cx+span, cy-inner, top))
    _append_box(vertices, triangles, (cx-span, cy+inner, bottom),
                (cx+span, cy+outer, top))


def narrow_pyramid_stairs_terrain(terrain, step_height, tread_depth=0.25, num_steps=8,
                                   platform_size=3., nosing_depth=0.03,
                                   nosing_height=0.035):
    """Generate an eight-level square stair mesh with true overhanging nosings."""
    step_height_units = int(step_height / terrain.vertical_scale)
    if step_height_units == 0:
        raise ValueError("narrow stair step height must be at least one vertical sample")
    actual_step_height = step_height_units * terrain.vertical_scale
    tile_length = terrain.length * terrain.horizontal_scale
    tile_width = terrain.width * terrain.horizontal_scale
    platform_half = platform_size / 2.
    stair_outer_half = platform_half + num_steps * tread_depth
    if 2. * stair_outer_half > min(tile_length, tile_width):
        raise ValueError("narrow stairs do not fit inside the terrain tile")

    direction = 1. if actual_step_height > 0. else -1.
    base_height = min(0., num_steps * actual_step_height)
    terrain.height_field_raw.fill(int(round(base_height / terrain.vertical_scale)))
    vertices = []
    triangles = []
    center = (tile_length / 2., tile_width / 2.)
    tile_half = min(tile_length, tile_width) / 2.

    if direction > 0.:
        for level in range(1, num_steps+1):
            outer_half = platform_half + (num_steps-level+1) * tread_depth
            inner_half = outer_half - tread_depth
            _append_square_annulus(vertices, triangles, center, outer_half, inner_half,
                                   base_height, level * actual_step_height)
        _append_box(vertices, triangles,
                    (center[0]-platform_half, center[1]-platform_half, base_height),
                    (center[0]+platform_half, center[1]+platform_half,
                     num_steps * actual_step_height))
    else:
        _append_square_annulus(vertices, triangles, center, tile_half, stair_outer_half,
                               base_height, 0.)
        for level in range(1, num_steps):
            outer_half = platform_half + (num_steps-level+1) * tread_depth
            inner_half = outer_half - tread_depth
            _append_square_annulus(vertices, triangles, center, outer_half, inner_half,
                                   base_height, level * actual_step_height)

    for level in range(1, num_steps+1):
        boundary_half = platform_half + (num_steps-level+1) * tread_depth
        high_level = level if direction > 0. else level-1
        _append_square_nosing(vertices, triangles, center, boundary_half,
                              high_level * actual_step_height, nosing_depth,
                              nosing_height, direction > 0.)

    terrain.narrow_stair_vertices = np.concatenate(vertices, axis=0)
    terrain.narrow_stair_triangles = np.concatenate(triangles, axis=0)
    terrain.narrow_stair_step_height = actual_step_height
    terrain.narrow_stair_platform_height = num_steps * actual_step_height
    return terrain


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        if cfg.narrow_stairs_enabled and self.type != "trimesh":
            raise ValueError("narrow stairs require terrain.mesh_type='trimesh'")
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.narrow_stair_mask = np.zeros((cfg.num_rows, cfg.num_cols), dtype=np.bool_)
        self.narrow_stair_step_heights = np.zeros((cfg.num_rows, cfg.num_cols), dtype=np.float32)
        self.extra_vertices = []
        self.extra_triangles = []
        self.extra_vertex_count = 0

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
            if self.extra_vertices:
                extra_vertices = np.concatenate(self.extra_vertices, axis=0)
                extra_triangles = np.concatenate(self.extra_triangles, axis=0) + self.vertices.shape[0]
                self.vertices = np.concatenate((self.vertices, extra_vertices), axis=0)
                self.triangles = np.concatenate((self.triangles, extra_triangles), axis=0)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty, i, j)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty, i, j)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty, row, col):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        selection = self.cfg.terrain_selection
        if selection != "mixed":
            if selection == "flat":
                return terrain
            step_height = abs(self.cfg.selected_stair_height)
            if selection.endswith("_down"):
                step_height *= -1
            if selection.startswith("narrow_stairs_"):
                narrow_pyramid_stairs_terrain(
                    terrain, step_height, self.cfg.narrow_stair_tread_depth,
                    self.cfg.narrow_stair_steps, 3., self.cfg.stair_nosing_depth,
                    self.cfg.stair_nosing_height)
            elif selection.startswith("stairs_"):
                terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31,
                                                      step_height=step_height, platform_size=3.)
            else:
                raise ValueError(f"unknown terrain selection: {selection}")
            return terrain
        slope = difficulty * 0.4
        step_height = 0.05 + 0.15 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            if self.cfg.narrow_stairs_enabled and (row + col) % 2 == 0:
                narrow_pyramid_stairs_terrain(
                    terrain, step_height, self.cfg.narrow_stair_tread_depth,
                    self.cfg.narrow_stair_steps, 3., self.cfg.stair_nosing_depth,
                    self.cfg.stair_nosing_height)
            else:
                terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        if hasattr(terrain, "narrow_stair_vertices"):
            offset = np.array([start_x * terrain.horizontal_scale,
                               start_y * terrain.horizontal_scale, 0.], dtype=np.float32)
            self.extra_vertices.append(terrain.narrow_stair_vertices + offset)
            self.extra_triangles.append(terrain.narrow_stair_triangles + self.extra_vertex_count)
            self.extra_vertex_count += terrain.narrow_stair_vertices.shape[0]
            self.narrow_stair_mask[i, j] = True
            self.narrow_stair_step_heights[i, j] = terrain.narrow_stair_step_height

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        if hasattr(terrain, "narrow_stair_platform_height"):
            env_origin_z = terrain.narrow_stair_platform_height
        else:
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
