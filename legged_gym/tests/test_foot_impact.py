import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import isaacgym
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.go2.go2_config import GO2RoughCfg


def make_task(physics_dt=0.005, decimation=4):
    task = LeggedRobot.__new__(LeggedRobot)
    task.cfg = GO2RoughCfg()
    task.cfg.control.decimation = decimation
    task.sim_params = SimpleNamespace(dt=physics_dt)
    task.dt = physics_dt * decimation
    task.num_envs = 2
    task.device = 'cpu'
    task.feet_indices = torch.arange(4)
    task.contact_forces = torch.zeros(2, 4, 3)
    task.rigid_body_states = torch.zeros(2, 4, 13)
    task.track_foot_impacts = True
    task.impact_contact_gap_steps = math.ceil(task.cfg.rewards.impact_contact_gap_s / physics_dt)
    task.impact_window_length_steps = math.ceil(task.cfg.rewards.impact_window_s / physics_dt)
    task.impact_air_steps = torch.zeros(2, 4, dtype=torch.long)
    task.impact_window_steps = torch.zeros_like(task.impact_air_steps)
    task.impact_last_air_velocity = torch.zeros(2, 4)
    task.foot_impact_velocity_cost = torch.zeros(2)
    task.foot_impact_force_cost = torch.zeros(2)
    return task


def set_sample(task, force_z, velocity_z=0., foot=3):
    task.contact_forces[:, :, 2] = 40.
    task.contact_forces[:, foot, 2] = force_z
    task.rigid_body_states[:, foot, 9] = velocity_z


def sample(task, force_z, velocity_z=0., foot=3):
    set_sample(task, force_z, velocity_z, foot)
    task._update_foot_impact_state()


def swing(task, velocity_z=-2.86, foot=3):
    for _ in range(task.impact_contact_gap_steps):
        sample(task, 0., velocity_z, foot)


class FootImpactTest(unittest.TestCase):
    def test_precontact_velocity_survives_collision_deceleration(self):
        task = make_task()
        swing(task)
        sample(task, 500., 0.)
        expected = torch.full((2,), (2.86 - 1.2) ** 2)
        torch.testing.assert_close(task._reward_foot_impact_velocity(), expected)
        sample(task, 500., 0.2)
        torch.testing.assert_close(task._reward_foot_impact_velocity(), expected)

    def test_below_threshold_and_upward_touchdown_have_no_speed_cost(self):
        for velocity in (-1., 2.):
            task = make_task()
            swing(task, velocity)
            sample(task, 249.)
            torch.testing.assert_close(task._reward_foot_impact_velocity(), torch.zeros(2))
            torch.testing.assert_close(task._reward_foot_impact_force(), torch.zeros(2))

    def test_window_expires_after_80_ms_and_does_not_penalize_late_support(self):
        task = make_task()
        swing(task)
        for _ in range(16):
            sample(task, 500.)
        torch.testing.assert_close(task._reward_foot_impact_force(), torch.full((2,), 4.))
        self.assertEqual(task.impact_window_steps.count_nonzero().item(), 0)
        for _ in range(20):
            sample(task, 500.)
        torch.testing.assert_close(task._reward_foot_impact_force(), torch.full((2,), 4.))

    def test_short_contact_gap_does_not_restart_window_or_repeat_speed_cost(self):
        task = make_task()
        swing(task)
        sample(task, 500.)
        task.foot_impact_velocity_cost.zero_()
        sample(task, 0., -3.)
        sample(task, 500.)
        torch.testing.assert_close(task._reward_foot_impact_velocity(), torch.zeros(2))
        self.assertTrue(torch.all(task.impact_window_steps[:, 3] == 13))

    def test_20_ms_airborne_rearms_touchdown(self):
        task = make_task()
        swing(task)
        sample(task, 500.)
        task.foot_impact_velocity_cost.zero_()
        swing(task, -2.)
        sample(task, 500.)
        torch.testing.assert_close(task._reward_foot_impact_velocity(), torch.full((2,), 0.8 ** 2))
        self.assertTrue(torch.all(task.impact_window_steps[:, 3] == 15))

    def test_existing_support_does_not_create_a_touchdown(self):
        task = make_task()
        for _ in range(20):
            sample(task, 500., -3.)
        torch.testing.assert_close(task._reward_foot_impact_velocity(), torch.zeros(2))
        torch.testing.assert_close(task._reward_foot_impact_force(), torch.zeros(2))

    def test_force_cost_is_integrated_once_across_timesteps_and_decimations(self):
        for physics_dt, decimation in ((0.005, 4), (0.0025, 8), (0.005, 2)):
            with self.subTest(physics_dt=physics_dt, decimation=decimation):
                task = make_task(physics_dt, decimation)
                task.reward_scales = {'foot_impact_force': task.cfg.rewards.scales.foot_impact_force}
                task.reward_curriculum_scales = {}
                task.rew_buf = torch.zeros(2)
                task._prepare_reward_function()
                swing(task)
                total_reward = torch.zeros(2)
                for _ in range(task.impact_window_length_steps // decimation):
                    task.foot_impact_force_cost.zero_()
                    for _ in range(decimation):
                        sample(task, 500.)
                    task.compute_reward()
                    total_reward += task.rew_buf
                torch.testing.assert_close(total_reward, torch.full((2,), -0.2 * 0.08))

    def test_each_foot_uses_the_same_penalty(self):
        for foot in range(4):
            with self.subTest(foot=foot):
                task = make_task()
                swing(task, foot=foot)
                sample(task, 500., foot=foot)
                torch.testing.assert_close(task._reward_foot_impact_force(), torch.full((2,), 0.25))
                torch.testing.assert_close(task._reward_foot_impact_velocity(), torch.full((2,), 1.66 ** 2))

    def test_step_captures_5_ms_pulse_missing_at_end_of_control_period(self):
        task = make_task()
        swing(task)
        task.cfg.domain_rand.randomize_action_delay = False
        task.cfg.domain_rand.randomize_motor_strength = False
        task.num_actions = 12
        task.sim = None
        task.render = lambda: None
        task._compute_torques = lambda actions: torch.zeros_like(actions)
        task.torques = torch.zeros(2, 12)
        task.post_physics_step = lambda: None
        task.obs_buf = torch.zeros(2, 45)
        task.privileged_obs_buf = None
        task.rew_buf = torch.zeros(2)
        task.reset_buf = torch.zeros(2, dtype=torch.bool)
        task.extras = {}
        task.foot_impact_velocity_cost.fill_(99.)
        task.foot_impact_force_cost.fill_(99.)
        pulses = iter([500., 0., 0., 0.])
        task.gym = Mock()
        task.gym.simulate.side_effect = lambda sim: set_sample(task, next(pulses))
        with patch('legged_gym.envs.base.legged_robot.gymtorch.unwrap_tensor', side_effect=lambda tensor: tensor):
            task.step(torch.zeros(2, 12))
            torch.testing.assert_close(task._reward_foot_impact_force(), torch.full((2,), 0.25))
            torch.testing.assert_close(task._reward_foot_impact_velocity(), torch.full((2,), 1.66 ** 2))
            self.assertTrue(torch.all(task.contact_forces[:, 3, 2] == 0.))
            task.gym.simulate.side_effect = lambda sim: set_sample(task, 40.)
            task.step(torch.zeros(2, 12))
        torch.testing.assert_close(task._reward_foot_impact_force(), torch.zeros(2))
        torch.testing.assert_close(task._reward_foot_impact_velocity(), torch.zeros(2))

    def test_reset_clears_only_selected_environment_and_cannot_reuse_old_velocity(self):
        task = make_task()
        for name in ('randomize_motor_strength', 'randomize_motor_zero_offset', 'randomize_pd_gains'):
            setattr(task.cfg.domain_rand, name, False)
        task.cfg.terrain.curriculum = False
        task.cfg.commands.curriculum = False
        task.cfg.commands.zero_command_curriculum = None
        task.cfg.env.send_timeouts = False
        task._reset_dofs = lambda ids: None
        task._reset_root_states = lambda ids: None
        task._resample_commands = lambda ids: None
        for name in ('actions', 'last_actions', 'last_last_actions', 'last_dof_vel'):
            setattr(task, name, torch.ones(2, 12))
        for name in ('feet_air_time', 'feet_contact_time', 'feet_air_time_on_contact'):
            setattr(task, name, torch.ones(2, 4))
        for name in ('last_contacts', 'filtered_foot_contacts', 'first_foot_contacts'):
            setattr(task, name, torch.ones(2, 4, dtype=torch.bool))
        task.phase_foot_reference = torch.ones(2, 4, 3)
        task.phase_foot_reference_initialized = torch.ones(2, dtype=torch.bool)
        task.episode_length_buf = torch.ones(2, dtype=torch.long)
        task.reset_buf = torch.zeros(2, dtype=torch.bool)
        task.extras = {}
        task.episode_sums = {}
        task.reward_curriculum_scales = {}
        task.impact_air_steps.fill_(4)
        task.impact_window_steps.fill_(15)
        task.impact_last_air_velocity.fill_(-3.)
        task.foot_impact_velocity_cost.fill_(5.)
        task.foot_impact_force_cost.fill_(6.)
        task.reset_idx(torch.tensor([0]))
        for name, previous in (('impact_air_steps', 4), ('impact_window_steps', 15),
                               ('impact_last_air_velocity', -3.), ('foot_impact_velocity_cost', 5.),
                               ('foot_impact_force_cost', 6.)):
            values = getattr(task, name)
            self.assertTrue(torch.all(values[0] == 0))
            self.assertTrue(torch.all(values[1] == previous))
        sample(task, 500.)
        self.assertEqual(task._reward_foot_impact_velocity()[0].item(), 0.)
        self.assertEqual(task._reward_foot_impact_force()[0].item(), 0.)


if __name__ == '__main__':
    unittest.main()
