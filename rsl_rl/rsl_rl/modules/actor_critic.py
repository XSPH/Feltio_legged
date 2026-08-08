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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .estimator import Estimator

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        history_length,
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        teacher_encoder_hidden_dims=[512, 256],
                        him_encoder_hidden_dims=[512, 256, 128],
                        him_target_hidden_dims=[128, 64],
                        activation='elu',
                        init_noise_std=1.0,
                        latent_dim=16,
                        num_prototypes=16,
                        him_temperature=3.0,
                        sinkhorn_epsilon=0.05,
                        sinkhorn_iterations=3,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation_name = activation
        activation = get_activation(activation)

        self.num_actor_obs = num_actor_obs
        self.history_length = history_length
        self.latent_dim = latent_dim
        self.context_dim = 3 + latent_dim

        mlp_input_dim_t = num_critic_obs
        mlp_input_dim_a = num_actor_obs + self.context_dim
        mlp_input_dim_c = num_critic_obs + self.context_dim

        # Teacher encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(mlp_input_dim_t, teacher_encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(teacher_encoder_hidden_dims)):
            if l == len(teacher_encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(teacher_encoder_hidden_dims[l], latent_dim))
                encoder_layers.append(L2Norm())

            else:
                encoder_layers.append(nn.Linear(teacher_encoder_hidden_dims[l], teacher_encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.teacher_encoder = nn.Sequential(*encoder_layers)

        # HIM student estimator
        self.estimator = Estimator(
            temporal_steps=history_length,
            num_one_step_obs=num_actor_obs,
            response_dim=num_actor_obs,
            latent_dim=latent_dim,
            enc_hidden_dims=him_encoder_hidden_dims,
            tar_hidden_dims=him_target_hidden_dims,
            activation=activation_name,
            num_prototypes=num_prototypes,
            temperature=him_temperature,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
        )

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Teacher Encoder MLP: {self.teacher_encoder}")
        print(f"HIM Estimator: {self.estimator}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def encode_teacher(self, privileged_obs):
        velocity_start = self.num_actor_obs
        velocity_end = velocity_start + 3
        if privileged_obs.shape[-1] < velocity_end:
            raise ValueError(
                "privileged_obs does not contain the expected base velocity at "
                f"indices {velocity_start}:{velocity_end}"
            )
        velocity = privileged_obs[:, velocity_start:velocity_end]
        latent = self.teacher_encoder(privileged_obs)
        return torch.cat([velocity, latent], dim=-1)

    def encode_student(self, history):
        predicted_velocity, latent = self.estimator.encode_history(history)
        return torch.cat([predicted_velocity, latent], dim=-1)

    def update_distribution(self, observations, is_teacher, privileged_obs, history):
        if is_teacher:
            context = self.encode_teacher(privileged_obs)
        else:
            with torch.no_grad():
                context = self.encode_student(history)
        actor_input = torch.cat([observations, context], dim=-1)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, obs, privileged_obs, history, is_teacher, **kwargs):
        self.update_distribution(obs, is_teacher, privileged_obs, history)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs, history):
        context = self.encode_student(history)
        actor_input = torch.cat([obs, context], dim=-1)
        actions_mean = self.actor(actor_input)
        return actions_mean

    def evaluate(self, privileged_obs, history, is_teacher, **kwargs):
        if is_teacher:
            context = self.encode_teacher(privileged_obs)
        else:
            with torch.no_grad():
                context = self.encode_student(history)
        critic_input = torch.cat([privileged_obs, context.detach()], dim=-1)
        value = self.critic(critic_input)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

class L2Norm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x, p=2.0, dim=-1)
