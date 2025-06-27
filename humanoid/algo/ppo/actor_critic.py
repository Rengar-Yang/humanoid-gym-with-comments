# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
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
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self,  num_actor_obs, # Actor的维度，Actor神经网络（策略网络）第一层的输入维度, =frame_stack * num_single_obs
                        num_critic_obs, # Critic的维度，Critic神经网络（价值网络）的第一层的输入维度, = frame_stack * single_num_privileged_obs
                        num_actions, # 动作的维度，是Actor神经网络（策略网络）最后一层的输出维度，机器人有几个动作就有几个维度
                        actor_hidden_dims=[256, 256, 256],# Actor神经网络（策略网络）中间的层(隐藏层)的维度
                        critic_hidden_dims=[256, 256, 256], # Critic神经网络（价值网络）中间的层(隐藏层)的维度
                        base_lin_vel_hidden_dims=[128, 128], # BaseLinear网络中间的隐藏层的维度
                        init_noise_std=1.0, # 噪声标准差
                        activation = nn.ELU(), # 激活函数，把所有数据变成正数，引入非线性项
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        mlp_input_dim_a = num_actor_obs # Actor网络隐藏层（除了最后一层输出层的所有的层都是隐藏层）维度 = 第一层的输入维度
        mlp_input_dim_c = num_critic_obs # Critic网络隐藏层维度 = 第一层的输入维度
        # Policy
        actor_layers = [] # 初始化一个空的Actor网络层
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0])) # 第一层是输入层，输入维度是obs的维度，输出维度为actor_hidden_dims[0]
        actor_layers.append(activation) # 然后插入激活函数层
        for l in range(len(actor_hidden_dims)): # 存在3个隐藏层，用for循环全部插入在层后面
            if l == len(actor_hidden_dims) - 1: # 如果是最后一层，则是输出层，输入维度是actor_hidden_dims[l]，输出维度是action的数量
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else: # 不是最后一层，则和初始化时一样，插一个线性变化函数加一个激活函数
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation) # 每一个隐藏层都要包含一个激活函数
        self.actor = nn.Sequential(*actor_layers)

        # 若想增加或改变网络结构，在actor网络和critic网络中添加新的层，并修改对应的参数即可
        #===================================================自定义网络层========================================================#
        # Base vel network
        base_lin_vel_layers = []
        base_lin_vel_layers.append(nn.Linear(mlp_input_dim_a, base_lin_vel_hidden_dims[0]))
        base_lin_vel_layers.append(activation)
        for l in range(len(base_lin_vel_hidden_dims)):
            if l == len(base_lin_vel_hidden_dims) - 1:
                base_lin_vel_layers.append(nn.Linear(base_lin_vel_hidden_dims[l], 3)) # vel网络输出的是3维线速度，所以维度是3
            else:
                base_lin_vel_layers.append(nn.Linear(base_lin_vel_hidden_dims[l], base_lin_vel_hidden_dims[l + 1]))
                base_lin_vel_layers.append(activation)
        self.base_lin_vel = nn.Sequential(*base_lin_vel_layers)
        #=====================================================================================================================#

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1)) # critic网络输出的是价值，所以维度是1
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Lin vel MLP: {self.base_lin_vel}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        

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

    def update_distribution(self, observations): # 更新动作的概率分布
        mean = self.actor(observations) # actor网络输出的是动作的均值
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs): # 对动作的概率分布进行采样
        self.update_distribution(observations)
        action = self.distribution.sample() # 随机采样,得到具体动作
        base_lin_vel = self.base_get_lin_vel(observations) # 直接获取vel网络的输出
        return action, base_lin_vel
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs): # 推理critic网络
        value = self.critic(critic_observations) # 直接获取critic网络输出
        return value
    def base_get_lin_vel(self, observations): 
        base_lin_vel = self.base_lin_vel(observations)
        return base_lin_vel