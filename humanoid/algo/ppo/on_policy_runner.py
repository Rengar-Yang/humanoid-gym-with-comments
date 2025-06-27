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

import os
import time
import torch
import wandb
import statistics
from collections import deque
from datetime import datetime
from .ppo import PPO
from .actor_critic import ActorCritic
from humanoid.algo.vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter


class OnPolicyRunner:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.cfg = train_cfg["runner"] # 这些信息都是用户设置的，比如最大迭代次数，学习率等，可以在config文件中找到
        self.alg_cfg = train_cfg["algorithm"] # PPO算法的参数都是cfg文件决定的
        self.policy_cfg = train_cfg["policy"]
        self.all_cfg = train_cfg
        self.wandb_run_name = (
            datetime.now().strftime("%b%d_%H-%M-%S")
            + "_"
            + train_cfg["runner"]["experiment_name"]
            + "_"
            + train_cfg["runner"]["run_name"]
        )
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs # Critic的维度是特权观测值的维度
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])  # 默认ActorCritic，这个是在每个机器人的congig.py中的Runner类定义的
        actor_critic: ActorCritic = actor_critic_class( # 创建一个ActorCritic实例
            self.env.num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg # 传入用户配置中的参数
        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])  # 默认PPO，也是在每个机器人的congig.py中的Runner类定义的
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg) # 把ActorCritic模型传入PPO类中，并实例化PPO
        self.num_steps_per_env = self.cfg["num_steps_per_env"] 
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0 # 这个变量只会在一整轮训练结束后才更新一次，用于resume时记录总共训练次数的

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False): # 学习函数
        # initialize writer
        if self.log_dir is not None and self.writer is None: # 初始化wandb
            wandb.init(
                project="XBot",
                sync_tensorboard=True,
                name=self.wandb_run_name,
                config=self.all_cfg,
                mode="offline",
            )
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10) # 初始化tensorboard
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like( # 随机数，范围是[0, self.env.max_episode_length)]
                self.env.episode_length_buf, high=int(self.env.max_episode_length) # 随机化episode的长度 （episode是一次完整的交互回合）
            )
        obs = self.env.get_observations() # 获取观测值，这个函数在base_task里面定义的,得到的是env文件中的obs_buf变量
        privileged_obs = self.env.get_privileged_observations() # 获取特权观测值，意思是机器人无法获取的状态信息，但是仿真环境可以用的反馈数据
        critic_obs = privileged_obs if privileged_obs is not None else obs # Critic的观测值优先设置为特权观测值
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device) # 把数据转到GPU上
        self.alg.actor_critic.train() # 把模型设置为训练模式

        ep_infos = [] # 存储episode的信息
        rewbuffer = deque(maxlen=100) # 存储奖励
        lenbuffer = deque(maxlen=100) # 存储episode的长度
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device # 初始化当前episode的总奖励
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device # 初始化当前episode的长度
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations # 总迭代次数，意思是训练多少次
        for it in range(self.current_learning_iteration, tot_iter): # 根据迭代次数用一个for循环进行迭代，从当前索引到末尾，这个是主循环，训练过程中最外层的那个循环，这个循环结束了代码就停止运行了
            start = time.time()
            # Rollout
            with torch.inference_mode(): # 评估模式，这段代码不用计算梯度，提高效率。不是循环，只对作用范围内的代码起作用
                for i in range(self.num_steps_per_env): # 循环进行env_num次env的交互，这个时在一轮训练中不断让一个智能体与环境进行交互，每满num_steps_per_env次之后更新一次模型
                    actions = self.alg.act(obs, critic_obs) # 从模型中获取动作
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions) # 通过step函数得到执行完动作action后得到的各种数据，比如奖励等。这个函数在env文件里面定义的
                    critic_obs = privileged_obs if privileged_obs is not None else obs # Critic网络输入是特权观测值
                    obs, critic_obs, rewards, dones = ( # 把数据放入GPU
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos) # 把数据放入存储器

                    if self.log_dir is not None: 
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards # 计算这几次交互的总奖励
                        cur_episode_length += 1 
                        new_ids = (dones > 0).nonzero(as_tuple=False) # 获取当前episode结束的env的id
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist() # 把当前episode的总奖励放入队列
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist() # 把当前episode的长度放入队列
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time() # num_steps_per_env次交互结束
                collection_time = stop - start # 总消耗的时间

                self.env.course_gain *=self.env.course_ratio # 课程增益
                self.env.course_gain = min(20,self.env.course_gain)
                course_gain = self.env.course_gain

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs) # 计算回报和优势

            mean_value_loss, mean_surrogate_loss, sym_loss, mean_base_lin_vel_loss = self.alg.update() # 模型训练的关键，计算loss并更新模型参数
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals()) # 保存日志
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it))) # 到达一定次数后保存模型
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations # 更新当前迭代次数
        self.save(
            os.path.join(
                self.log_dir, "model_{}.pt".format(self.current_learning_iteration) # 保存模型
            )
        )

    def log(self, locs, width=90, pad=45): # 日志函数,会一直打印
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]: # 这里会自动加载自己设置的reward
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        # reward以外的变量，可以加在这里
        self.writer.add_scalar(
            "Loss/sym_loss", locs["sym_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Train/course_gain", locs["course_gain"], locs["it"]
        )
        self.writer.add_scalar("Loss/mean_base_lin_vel_loss", locs["mean_base_lin_vel_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_reward/time",
                statistics.mean(locs["rewbuffer"]),
                self.tot_time,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]),
                self.tot_time,
            )

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0: # 这里用于训练时终端实时打印
            log_string = ( # 包含一些总体数据
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'System loss:':>{pad}} {locs['sym_loss']:.4f}\n"""
                f"""{'Base vel loss:':>{pad}} {locs['mean_base_lin_vel_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string # 这个字符串包含了各个reward相关变量
        log_string += ( # 包含时间相关变量
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None): # 保存模型
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True): # 加载模型
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None): # 获取policy
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_inference_critic(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.evaluate
