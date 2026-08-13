from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import math
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_scale_shift
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        if self.cfg.domain_rand.randomize_action_delay:
            actions_start_decimation = torch.randint(
                0, self.cfg.control.decimation + 1, (self.num_envs, 1), device=self.device
            )
        for i in range(self.cfg.control.decimation):
            if self.cfg.domain_rand.randomize_action_delay:
                use_new_actions = (i >= actions_start_decimation).float()
                input_actions = (1.0 - use_new_actions) * self.last_actions + use_new_actions * self.actions
            else:
                input_actions = self.actions
            self.torques = self._compute_torques(input_actions).view(self.torques.shape)
            if self.cfg.domain_rand.randomize_motor_strength:
                self.torques *= self.motor_strengths
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.foot_positions = self.rigid_body_states[:, self.feet_indices, :3]
        self.foot_velocities = self.rigid_body_states[:, self.feet_indices, 7:10]
        self._update_foot_contact_state()

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # domain randomization at environment reset
        if self.cfg.domain_rand.randomize_motor_strength:
            rng = self.cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids] = torch_rand_float(
                rng[0], rng[1], (len(env_ids), self.num_actions), device=self.device
            )
        if self.cfg.domain_rand.randomize_motor_zero_offset:
            rng = self.cfg.domain_rand.motor_zero_offset_range
            self.motor_zero_offsets[env_ids] = torch_rand_float(
                rng[0], rng[1], (len(env_ids), self.num_actions), device=self.device
            )
        if self.cfg.domain_rand.randomize_pd_gains:
            stiffness_range = self.cfg.domain_rand.stiffness_multiplier_range
            damping_range = self.cfg.domain_rand.damping_multiplier_range
            self.p_gains_multiplier[env_ids] = torch_rand_float(
                stiffness_range[0], stiffness_range[1],
                (len(env_ids), self.num_actions), device=self.device
            )
            self.d_gains_multiplier[env_ids] = torch_rand_float(
                damping_range[0], damping_range[1],
                (len(env_ids), self.num_actions), device=self.device
            )

        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.feet_contact_time[env_ids] = 0.
        self.feet_air_time_on_contact[env_ids] = 0.
        self.last_contacts[env_ids] = False
        self.filtered_foot_contacts[env_ids] = False
        self.first_foot_contacts[env_ids] = False
        self.phase_foot_reference[env_ids] = 0.
        self.phase_foot_reference_initialized[env_ids] = False
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)

        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,-1.,1.) * self.obs_scales.height_measurements
        self.foot_contact_forces = self._get_normalized_foot_contact_forces()
        self.privileged_obs_buf = torch.cat((
                                    self.obs_buf,
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.foot_contact_forces,
                                    heights
                                    ), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_normalized_foot_contact_forces(self):
        """Return clipped foot contact forces in the base frame."""
        self.contact_force_xy_scale, self.contact_force_xy_shift = get_scale_shift(self.cfg.normalization.contact_force_xy_range)
        self.contact_force_z_scale, self.contact_force_z_shift = get_scale_shift(self.cfg.normalization.contact_force_z_range)
        num_feet = len(self.feet_indices)
        foot_forces = self.contact_forces[:, self.feet_indices, :]
        base_quaternions = self.base_quat.unsqueeze(1).repeat(1, num_feet, 1)
        foot_forces = quat_rotate_inverse(base_quaternions.reshape(-1, 4),foot_forces.reshape(-1, 3),).view(self.num_envs, num_feet, 3)
        normalized_forces = torch.cat((
            (foot_forces[..., :2] - self.contact_force_xy_shift) * self.contact_force_xy_scale,
            (foot_forces[..., 2:] - self.contact_force_z_shift) * self.contact_force_z_scale,
        ), dim=-1)
        return torch.clip(normalized_forces, -1., 1.).reshape(self.num_envs, -1)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            restitution = np.random.uniform(*self.cfg.domain_rand.restitution_range)
            for shape_prop in props:
                shape_prop.restitution = restitution
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.multiplied_link_mass_range
            mass_multipliers = np.random.uniform(rng[0], rng[1], len(props) - 1)
            for body_prop, multiplier in zip(props[1:], mass_multipliers):
                body_prop.mass *= multiplier

        if self.cfg.domain_rand.randomize_base_com:
            rng = self.cfg.domain_rand.added_base_com_range
            com_offset = np.random.uniform(rng[0], rng[1], 3)
            props[0].com += gymapi.Vec3(*(float(value) for value in com_offset))
        return props

    def _update_foot_contact_state(self):
        """统一更新奖励函数使用的足端接触状态和连续计时。"""
        # PhysX 在网格地形上的单帧接触检测可能抖动，因此将当前帧和上一帧的原始接触取或，
        # 得到更稳定的 filtered_contacts。这里每个仿真步只更新一次，避免奖励计算顺序影响计时。
        contacts = (self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.foot_contact_force_threshold)
        filtered_contacts = torch.logical_or(contacts, self.last_contacts)
        next_air_time = self.feet_air_time + self.dt
        self.filtered_foot_contacts[:] = filtered_contacts
        # first_foot_contacts 只在脚结束腾空、首次重新接触地面时为 True。
        self.first_foot_contacts[:] = torch.logical_and(self.feet_air_time > 0., filtered_contacts)
        # 先保存本次触地前的完整腾空时长，再把已经触地的脚的腾空计时清零。
        self.feet_air_time_on_contact[:] = next_air_time
        self.feet_air_time[:] = torch.where(filtered_contacts, torch.zeros_like(self.feet_air_time), next_air_time)
        # 接触计时与腾空计时互斥：接触时累加，离地后立即清零。
        self.feet_contact_time[:] = torch.where(filtered_contacts, self.feet_contact_time + self.dt, torch.zeros_like(self.feet_contact_time))
        self.last_contacts[:] = contacts
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *=self.cfg.control.hip_reduction
        control_type = self.cfg.control.control_type
        p_gains = self.p_gains * self.p_gains_multiplier
        d_gains = self.d_gains * self.d_gains_multiplier
        if control_type=="P":
            torques = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_zero_offsets) - d_gains * self.dof_vel
        elif control_type=="V":
            torques = p_gains * (actions_scaled - self.dof_vel) - d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        if len(push_env_ids) == 0:
            return
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_ang_vel = self.cfg.domain_rand.max_push_ang_vel
        self.root_states[push_env_ids, 7:9] = torch_rand_float(
            -max_vel, max_vel, (len(push_env_ids), 2), device=self.device
        ) # lin vel x/y
        self.root_states[push_env_ids, 10:13] = torch_rand_float(
            -max_ang_vel, max_ang_vel, (len(push_env_ids), 3), device=self.device
        ) # ang vel x/y/z
        
        env_ids_int32 = push_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[45:] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.base_pos = self.root_states[:, :3]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        self.foot_positions = self.rigid_body_states[:, self.feet_indices, :3]
        self.foot_velocities = self.rigid_body_states[:, self.feet_indices, 7:10]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_zero_offsets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains_multiplier = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains_multiplier = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_contact_time = torch.zeros_like(self.feet_air_time)
        self.feet_air_time_on_contact = torch.zeros_like(self.feet_air_time)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.filtered_foot_contacts = torch.zeros_like(self.last_contacts)
        self.first_foot_contacts = torch.zeros_like(self.last_contacts)
        self.phase_foot_reference = torch.zeros(self.num_envs, len(self.feet_indices), 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.phase_foot_reference_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        gait_synced_feet = self.cfg.rewards.gait_synced_feet
        if gait_synced_feet:
            if len(gait_synced_feet) != 2 or any(len(pair) != 2 for pair in gait_synced_feet):
                raise ValueError("feet_gait requires exactly two pairs of synchronized feet")
            feet_name_to_index = {name: index for index, name in enumerate(self.feet_names)}
            self.gait_feet_indices = torch.tensor(
                [[feet_name_to_index[name] for name in pair] for pair in gait_synced_feet],
                dtype=torch.long, device=self.device, requires_grad=False)
        else:
            self.gait_feet_indices = torch.empty(0, 2, dtype=torch.long, device=self.device, requires_grad=False)
        joint_mirror_groups = self.cfg.rewards.joint_mirror_joint_groups
        if joint_mirror_groups:
            dof_name_to_index = {name: index for index, name in enumerate(self.dof_names)}
            self.joint_mirror_indices = torch.tensor(
                [[[dof_name_to_index[name] for name in joint_names] for joint_names in group]
                 for group in joint_mirror_groups],
                dtype=torch.long, device=self.device, requires_grad=False)
            self.joint_mirror_signs = torch.tensor(
                self.cfg.rewards.joint_mirror_signs,
                dtype=torch.float, device=self.device, requires_grad=False)
            if self.joint_mirror_indices.shape[1] != 2 or self.joint_mirror_indices.shape[2] != len(self.joint_mirror_signs):
                raise ValueError("joint_mirror groups and signs have incompatible shapes")
        else:
            self.joint_mirror_indices = torch.empty(0, 2, 0, dtype=torch.long, device=self.device, requires_grad=False)
            self.joint_mirror_signs = torch.empty(0, dtype=torch.float, device=self.device, requires_grad=False)
        self.phase_foot_trajectory_phase_offsets = torch.tensor(
            self.cfg.rewards.phase_foot_trajectory_phase_offsets,
            dtype=torch.float, device=self.device, requires_grad=False)
        if len(self.phase_foot_trajectory_phase_offsets) not in (0, len(self.feet_indices)):
            raise ValueError("phase_foot_trajectory_phase_offsets must match the number of feet")

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        self.feet_names = feet_names
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_foot_heights(self):
        """Sample terrain heights directly below the feet."""
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros_like(self.foot_positions[..., 2])

        if self.cfg.terrain.mesh_type in ("none", None):
            raise NameError("Can't measure foot height without terrain")

        points = (
            self.foot_positions[..., :2] + self.terrain.cfg.border_size
        ) / self.terrain.cfg.horizontal_scale

        px = points[..., 0].long().reshape(-1)
        py = points[..., 1].long().reshape(-1)
        px = px.clamp(0, self.height_samples.shape[0] - 2)
        py = py.clamp(0, self.height_samples.shape[1] - 2)

        h00 = self.height_samples[px, py]
        h10 = self.height_samples[px + 1, py]
        h01 = self.height_samples[px, py + 1]
        heights = torch.minimum(torch.minimum(h00, h10), h01)

        return (
            heights.view(self.num_envs, len(self.feet_indices))
            * self.terrain.cfg.vertical_scale
        )

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # print("base height", base_height.mean().item())
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        """在有平移指令时，根据落地前腾空时间奖励较长的步幅。"""
        # 仅在首次触地帧结算；腾空 0.5 s 为零点，更长为正奖励，更短为负奖励。
        rew_airTime = torch.sum((self.feet_air_time_on_contact - 0.5) * self.first_foot_contacts, dim=1) # reward only on first contact with the ground
        # 原地站立时关闭该项，防止策略为了刷腾空时间而无指令踏步。
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        return rew_airTime

    def _reward_feet_gait(self):
        """鼓励配置中的同相足同步、两组足交替，形成稳定步态。"""
        pairs = self.gait_feet_indices
        max_error_sq = self.cfg.rewards.gait_max_error ** 2
        error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # 同一组中的两只脚应同时腾空、同时支撑，因此比较各自的腾空和接触持续时间。
        for foot_0, foot_1 in (pairs[0], pairs[1]):
            error += torch.clamp(torch.square(self.feet_air_time[:, foot_0] - self.feet_air_time[:, foot_1]), max=max_error_sq)
            error += torch.clamp(torch.square(self.feet_contact_time[:, foot_0] - self.feet_contact_time[:, foot_1]), max=max_error_sq)
        # 两组足应反相：一组的腾空持续时间应接近另一组的接触持续时间，反之亦然。
        asynchronous_pairs = (
            (pairs[0, 0], pairs[1, 0]),
            (pairs[0, 1], pairs[1, 1]),
            (pairs[0, 0], pairs[1, 1]),
            (pairs[1, 0], pairs[0, 1]))
        for foot_0, foot_1 in asynchronous_pairs:
            error += torch.clamp(torch.square(self.feet_air_time[:, foot_0] - self.feet_contact_time[:, foot_1]), max=max_error_sq)
            error += torch.clamp(torch.square(self.feet_contact_time[:, foot_0] - self.feet_air_time[:, foot_1]), max=max_error_sq)
        # 只有收到运动指令或机器人已经在移动时才约束步态，避免干扰静止站立。
        command_active = torch.norm(self.commands[:, :3], dim=1) > self.cfg.rewards.command_threshold
        body_moving = torch.norm(self.base_lin_vel[:, :2], dim=1) > self.cfg.rewards.gait_velocity_threshold
        # 误差越小越接近 1；gait_sigma 越小，对不同步越敏感。
        return torch.exp(-error / self.cfg.rewards.gait_sigma) * torch.logical_or(command_active, body_moving)

    def _reward_joint_mirror(self):
        """计算配对腿相对默认关节角的镜像误差，配合负权重抑制不对称姿态。"""
        # 比较相对默认站姿的偏移，而非直接比较绝对关节角，避免破坏 Go2 的默认构型。
        joint_pos_relative = self.dof_pos - self.default_dof_pos
        left = joint_pos_relative[:, self.joint_mirror_indices[:, 0, :]]
        right = joint_pos_relative[:, self.joint_mirror_indices[:, 1, :]]
        # signs 指定各关节镜像后的方向关系：髋外展通常反号，大腿和小腿通常同号。
        error = torch.square(left - self.joint_mirror_signs.view(1, 1, -1) * right)
        # 此处返回非负误差；配置中的负 scale 将它转换为惩罚。
        return torch.mean(torch.sum(error, dim=2), dim=1)

    def _reward_feet_slide(self):
        """统计支撑脚的水平滑动速度，配合负权重惩罚打滑。"""
        # 只计算稳定接触的脚；腾空脚的正常摆动速度不会被惩罚。
        feet_speed_xy = torch.norm(self.foot_velocities[:, :, :2], dim=2)
        return torch.sum(feet_speed_xy * self.filtered_foot_contacts.float(), dim=1)

    def _reward_foot_impact_velocity(self):
        """统计首次触地时超过阈值的向下速度，配合负权重减轻落脚冲击。"""
        # 低于 impact_speed_threshold 的轻微向下速度不计入，超出部分使用平方惩罚。
        downward_speed = torch.clamp(-self.foot_velocities[:, :, 2] - self.cfg.rewards.impact_speed_threshold, min=0.)
        return torch.sum(self.first_foot_contacts.float() * torch.square(downward_speed), dim=1)

    def _reward_feet_contact_without_cmd(self):
        """零运动指令时按接触脚数量给奖励，鼓励机器人安稳四脚站立。"""
        command_is_zero = torch.norm(self.commands[:, :3], dim=1) < self.cfg.rewards.command_threshold
        # 有运动指令时该项为零，不会阻碍正常抬脚行走。
        return torch.sum(self.filtered_foot_contacts.float(), dim=1) * command_is_zero

    def _reward_phase_foot_trajectory_exp(self):
        """在机身坐标系中跟踪按相位生成的足端轨迹，并以指数形式给奖励。"""
        num_feet = len(self.feet_indices)
        cycle_time = self.cfg.rewards.phase_foot_trajectory_cycle_time
        stance_ratio = self.cfg.rewards.phase_foot_trajectory_stance_ratio
        horizontal_span = self.cfg.rewards.phase_foot_trajectory_horizontal_span
        swing_height = self.cfg.rewards.phase_foot_trajectory_swing_height
        # 每只脚使用同一个周期，并通过 phase_offsets 设置同相或反相关系。
        phase_time = self.episode_length_buf.float() * self.dt
        phase = torch.remainder(phase_time.unsqueeze(1) / cycle_time + self.phase_foot_trajectory_phase_offsets.unsqueeze(0), 1.)
        stance_mask = phase < stance_ratio
        # 支撑相：参考足端沿机身 x 轴从前向后匀速移动，z 方向保持不抬脚。
        stance_phase = phase / stance_ratio
        q_stance = horizontal_span * (1. - 2. * stance_phase)
        dq_stance = torch.full_like(phase, -2. * horizontal_span / (stance_ratio * cycle_time))
        # 摆动相：用五阶 Bezier 曲线从后向前摆腿，中段抬高至 swing_height。
        swing_phase = torch.clamp((phase - stance_ratio) / (1. - stance_ratio), 0., 1.)
        control_points = torch.tensor(
            [
                [-horizontal_span, 0.],
                [-0.95 * horizontal_span, 0.80 * swing_height],
                [-0.55 * horizontal_span, swing_height],
                [0.55 * horizontal_span, swing_height],
                [0.95 * horizontal_span, 0.80 * swing_height],
                [horizontal_span, 0.],
            ],
            dtype=torch.float, device=self.device, requires_grad=False)
        position_weights = torch.stack(
            [math.comb(5, index) * (1. - swing_phase) ** (5 - index) * swing_phase ** index for index in range(6)], dim=-1)
        swing_position = torch.matmul(position_weights, control_points)
        derivative_points = 5. * (control_points[1:] - control_points[:-1])
        derivative_weights = torch.stack(
            [math.comb(4, index) * (1. - swing_phase) ** (4 - index) * swing_phase ** index for index in range(5)], dim=-1)
        swing_velocity = torch.matmul(derivative_weights, derivative_points) / ((1. - stance_ratio) * cycle_time)
        # 根据当前相位拼出期望位置偏移 (x, y, z) 和期望速度；y 方向不施加摆动轨迹。
        q = torch.where(stance_mask, q_stance, swing_position[:, :, 0])
        z = torch.where(stance_mask, torch.zeros_like(phase), swing_position[:, :, 1])
        dq = torch.where(stance_mask, dq_stance, swing_velocity[:, :, 0])
        dz = torch.where(stance_mask, torch.zeros_like(phase), swing_velocity[:, :, 1])
        trajectory_offset = torch.stack((q, torch.zeros_like(q), z), dim=2)
        reference_velocity = torch.stack((dq, torch.zeros_like(dq), dz), dim=2)
        base_quat = self.base_quat.unsqueeze(1).expand(-1, num_feet, -1).reshape(-1, 4)
        relative_foot_position = self.foot_positions - self.root_states[:, :3].unsqueeze(1)
        relative_foot_velocity = self.foot_velocities - self.root_states[:, 7:10].unsqueeze(1)
        # 将世界坐标中的足端位置和速度转换到机身坐标系，使奖励不受机器人朝向影响。
        foot_position_base = quat_rotate_inverse(base_quat, relative_foot_position.reshape(-1, 3)).view(self.num_envs, num_feet, 3)
        foot_velocity_base = quat_rotate_inverse(base_quat, relative_foot_velocity.reshape(-1, 3)).view(self.num_envs, num_feet, 3)
        # 每回合首次计算时记录足端基准点，之后只要求跟踪相对于该基准点的周期偏移。
        uninitialized = ~self.phase_foot_reference_initialized
        self.phase_foot_reference[uninitialized] = foot_position_base[uninitialized] - trajectory_offset[uninitialized]
        self.phase_foot_reference_initialized[uninitialized] = True
        reference_position = self.phase_foot_reference + trajectory_offset
        # 位置误差始终参与；速度误差可通过 velocity_weight 调节，设为 0 时完全关闭。
        position_error = torch.sum(torch.square(foot_position_base - reference_position), dim=(1, 2))
        velocity_error = torch.sum(torch.square(foot_velocity_base - reference_velocity), dim=(1, 2))
        total_error = position_error + self.cfg.rewards.phase_foot_trajectory_velocity_weight * velocity_error
        # 误差为 0 时奖励为 1，误差增大后指数衰减；仅在存在运动指令时启用。
        reward = torch.exp(-total_error / self.cfg.rewards.phase_foot_trajectory_std ** 2)
        command_active = torch.norm(self.commands[:, :3], dim=1) > self.cfg.rewards.command_threshold
        return reward * command_active
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_feet_regulation(self):
        self.foot_heights = (self.foot_positions[:, :, 2] - 0.022 - self._get_foot_heights()).clamp_min(0.0)
        height_tar = self.cfg.rewards.base_height_target * 0.025
        feet_xy_speed_sq = torch.sum(torch.square(self.foot_velocities[:, :, :2]), dim=-1)
        return torch.sum(torch.exp(-self.foot_heights / height_tar) * feet_xy_speed_sq,dim=1)

    def _reward_action_smoothness(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - 2*self.last_actions + self.last_last_actions), dim=1)

    def _reward_hip_default(self):
        hip_dof_names = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint']
        hip_dof_indices = [0, 3, 6, 9]
        hip_pos = self.dof_pos[:, hip_dof_indices]
        default_hip_pos = self.default_dof_pos[:, hip_dof_indices]
        return torch.sum(torch.abs(hip_pos - default_hip_pos), dim=1)

    def _reward_dof_power(self):
        # Penalize power consumption
        power = self.torques * self.dof_vel
        rew = torch.sum(torch.abs(power), dim=1)
        return rew
