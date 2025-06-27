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
import torch.optim as optim
import torch.nn.functional as F

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage

import numpy as np

class PPO: # 强化学习PPO算法的Python实现
    actor_critic: ActorCritic
    def __init__(self, # PPO的各项参数，这些值都是从cfg中读取的，这里不用改，在on police runner实例化PPO的时候可以看到
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu', # 定义运行设备
                 sym_loss = False,
                 obs_permutation = None,
                 act_permutation = None,
                 frame_stack = 0,
                 sym_coef = 1.0,
                 base_lin_vel_coef = 1.0
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate) # 优化器，一般都是用Adam优化器
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.base_lin_vel_coef = base_lin_vel_coef

        #对称性约束
        self.sym_loss = sym_loss
        self.sym_coef = sym_coef
        if self.sym_loss: # 对称性约束，把左右腿的差异作为系统loss，以减小左右腿的输出差异
            self.act_perm_mat = torch.zeros((len(act_permutation), len(act_permutation))).cuda()
            for i, perm in enumerate(act_permutation):
                self.act_perm_mat[int(abs(perm))][i] = np.sign(perm) # 生成动作向量的对称映射矩阵
            obs_permutation_stack = []
            for i in range(frame_stack):
                for p in obs_permutation:
                    obs_permutation_stack.append(np.sign(p)*(abs(p)+i*len(obs_permutation)))  
            self.obs_perm_mat = torch.zeros((len(obs_permutation_stack), len(obs_permutation_stack))).cuda()
            for i, perm in enumerate(obs_permutation_stack):
                self.obs_perm_mat[int(abs(perm))][i] = np.sign(perm)  # 生成观测向量的对称映射矩阵

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self): # 测试模式
        self.actor_critic.test() 
    
    def train_mode(self): # 训练模式
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs)[0].detach() # 使用actor_critic根据obs计算动作
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach() # 使用actor_critic根据critic_obs计算价值
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach() # 计算动作的对数（log）的概率，求梯度用的
        self.transition.action_mean = self.actor_critic.action_mean.detach() # 计算动作均值
        self.transition.action_sigma = self.actor_critic.action_std.detach() # 计算动作标准差
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions # 返回动作
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone() # 把reward复制到transition.rewards中
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition) # 把transition添加到storage中，便于后续计算loss的时候调用数据
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach() # 上一周期策略网络根据观测值得到的价值
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self): # 更新策略网络
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_base_lin_vel_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs) # batch生成器，从 rollout 缓冲区中分批提取数据进行训练
        for obs_batch, critic_obs_batch, base_lin_vel_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator: # 这是训练循环的主干，对每个 mini-batch 进行一次策略更新


                action, est_base_lin_vel = self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]) # actor和vel网络推理
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch) # 计算动作对数概率
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]) # 获得价值，critic网络的输出
                mu_batch = self.actor_critic.action_mean # 计算动作均值，算KL散度用的
                sigma_batch = self.actor_critic.action_std # 计算动作标准差，算KL散度用的
                entropy_batch = self.actor_critic.entropy # 计算熵，算loss用的

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive': # 动作的KL散度
                    with torch.inference_mode(): # 只计算前向传播，不更新参数
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1) # 计算KL散度的数学表达式
                        kl_mean = torch.mean(kl) # 得到KL散度的均值

                        if kl_mean > self.desired_kl * 2.0: # 动作的KL散度大于2倍目标值，则学习率减小，PPO的思想
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0: # 动作的KL散度小于0.5倍目标值，则学习率增大
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups: # 更新学习率
                            param_group['lr'] = self.learning_rate # 更新学习率


                # Surrogate loss，用于优化act，使得策略能够增大能够使advantage更大的动作的概率
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)) # 计算动作对数概率的比值，如果 ratio > 1：说明新策略更“喜欢”这个动作
                # 如果 advantage > 0:希望 ratio 越 大 越好（更信任新策略）
                # 如果 advantage < 0:希望 ratio 越 小 越好（避免选择这个动作）
                surrogate = -torch.squeeze(advantages_batch) * ratio # advantage大于0就说明当前动作更好，这段 Loss 让 Actor 优化动作概率的输出，使其更偏向优势更高的动作
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, # 计算动作对数概率的Surrogate loss
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean() # 裁减Surrogate loss，用于限制模型更新速度，缓慢更新有助于找到稳定的值

                # Value function loss，用于优化critic网络，使得critic网络的输出和return值之间的误差尽可能小
                if self.use_clipped_value_loss: # 是否使用截断值函数
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param) # 截断值函数
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                #Sym loss，对称性约束，用于优化act，使得动作的symmetry更平滑，左右腿的动作更对称
                sym_loss = 0 
                if self.sym_loss: # 对称性约束的输出作为loss
                    mirror_obs = torch.matmul(obs_batch,self.obs_perm_mat) # obs与对称映射矩阵相乘，得到镜像的obs
                    mirror_act = self.actor_critic.actor(mirror_obs) # 把镜像的obs输入网络，得到网络输出的镜像的action
                    m_mirror_act = torch.matmul(mirror_act,self.act_perm_mat) # action与对称映射矩阵相乘，得到真实的镜像的action
                    sym_loss = (mu_batch-m_mirror_act).pow(2).mean() # 计算镜像action与真实action之间的差异，作为loss

                # Base lin vel loss，线速度预测，用于根据obs预测出线速度
                base_lin_vel_loss = F.mse_loss(est_base_lin_vel, base_lin_vel_batch)

                # 总loss
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + self.sym_coef * sym_loss + self.base_lin_vel_coef * base_lin_vel_loss # 计算总损失

                # Gradient step
                self.optimizer.zero_grad() # 清除梯度
                loss.backward() # 反向传播,梯度下降降低Loss
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm) # 梯度裁剪，防止变化过大引起震荡
                self.optimizer.step() # 更新参数，这里会自动使用当前学习率对参数进行更新

                mean_value_loss += value_loss.item() 
                mean_surrogate_loss += surrogate_loss.item()
                mean_base_lin_vel_loss += base_lin_vel_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches # 更新次数
        mean_value_loss /= num_updates  # loss的平均值
        mean_surrogate_loss /= num_updates # surrogateloss的平均值
        mean_base_lin_vel_loss /= num_updates
        self.storage.clear()  # 清除数据

        return mean_value_loss, mean_surrogate_loss, sym_loss, mean_base_lin_vel_loss # 返回误差loss
