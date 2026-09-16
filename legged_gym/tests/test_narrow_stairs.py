import unittest
from types import SimpleNamespace

from isaacgym import terrain_utils
import numpy as np
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.terrain import Terrain, narrow_pyramid_stairs_terrain


def make_cfg():
    return SimpleNamespace(mesh_type="trimesh", terrain_length=8., terrain_width=8.,
                           terrain_proportions=[0.2, 0.2, 0.35, 0.25, 0.0],
                           num_rows=2, num_cols=4, horizontal_scale=0.1,
                           vertical_scale=0.005, border_size=0., curriculum=True,
                           selected=False, slope_treshold=0.75,
                           narrow_stairs_enabled=True, narrow_stair_tread_depth=0.25,
                           narrow_stair_steps=8, stair_nosing_depth=0.03,
                           stair_nosing_height=0.035, terrain_selection="mixed",
                           selected_stair_height=0.14)


class NarrowStairsTest(unittest.TestCase):
    def test_mesh_dimensions_and_directions(self):
        for step_height in (0.1, -0.1):
            terrain = terrain_utils.SubTerrain("terrain", width=80, length=80,
                                               vertical_scale=0.005, horizontal_scale=0.1)
            narrow_pyramid_stairs_terrain(terrain, step_height)
            self.assertEqual(terrain.narrow_stair_vertices.shape[1], 3)
            self.assertEqual(terrain.narrow_stair_triangles.shape[1], 3)
            self.assertLess(terrain.narrow_stair_triangles.max(), terrain.narrow_stair_vertices.shape[0])
            self.assertAlmostEqual(terrain.narrow_stair_platform_height, 8 * step_height)
            self.assertAlmostEqual(abs(terrain.narrow_stair_step_height), 0.1)

    def test_fixed_half_distribution(self):
        terrain = Terrain(make_cfg(), 8)
        expected = np.array([[False, False, True, False],
                             [False, False, False, True]])
        np.testing.assert_array_equal(terrain.narrow_stair_mask, expected)
        self.assertLess(terrain.triangles.max(), terrain.vertices.shape[0])

    def test_selected_narrow_stairs_fill_all_tiles(self):
        for selection, sign in (("narrow_stairs_up", 1), ("narrow_stairs_down", -1)):
            cfg = make_cfg()
            cfg.terrain_selection = selection
            terrain = Terrain(cfg, 8)
            self.assertTrue(terrain.narrow_stair_mask.all())
            self.assertTrue((np.sign(terrain.narrow_stair_step_heights) == sign).all())

    def test_analytic_height_includes_nosing(self):
        cfg = make_cfg()
        task = SimpleNamespace(
            terrain=SimpleNamespace(env_length=8., env_width=8., cfg=cfg),
            cfg=SimpleNamespace(terrain=cfg),
            narrow_stair_mask=torch.tensor([[True]]),
            narrow_stair_step_heights=torch.tensor([[0.1]]),
        )
        cfg.num_rows = 1
        cfg.num_cols = 1
        outer_edge = 4. + 3.5
        points = torch.tensor([[[outer_edge + 0.029, 4.],
                                [outer_edge + 0.031, 4.],
                                [4., 4.]]])
        heights = LeggedRobot._apply_narrow_stair_heights(task, points, torch.zeros(1, 3))
        torch.testing.assert_close(heights, torch.tensor([[0.1, 0., 0.8]]))


if __name__ == "__main__":
    unittest.main()
