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


import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import XBotLCfg
from humanoid.envs import D11Cfg
import torch
from joystick import JoyStickController
from humanoid.utils import get_args, task_registry
import matplotlib.pyplot as plt
import time

joystick_controller = JoyStickController()
# 力的大小
FORCE_MAGNITUDE = 50  # 牛顿
Plot_data = True

class cmd: # 指令，这里指令是固定的向前走
    vx = -0.25
    vy = 0.0
    dyaw = 0.0


def quaternion_to_euler_array(quat): # 将四元数转换为欧拉角
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data): # 从数据中提取出观察值
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double) # 获取关节角度
    # print(data.qpos)
    dq = data.qvel.astype(np.double) # 获取关节速度
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double) # 获取四元数
    r = R.from_quat(quat) # 将四元数转换为旋转矩阵
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame # velocity in the base frame
    omega = data.sensor('angular_velocity').data.astype(np.double) # 角速度
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double) # 加速度
    return (q, dq, quat, v, omega, gvec) # 返回关节角度，关节角速度，四元数，机器人线速度，机器人角速度，机器人加速度

def pd_control(target_q, q, kp, target_dq, dq, kd): # 经典的PD控制器
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg): # 运行mujoco仿真
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    if Plot_data:
        # 初始化图表
        plt.ion()  # 开启交互模式
        fig, axes = plt.subplots(2, 6, figsize=(15, 8))
        axes = axes.flatten()

        # 初始化曲线
        lines_target = []
        lines_q = []
        for i in range(12):
            ax = axes[i]
            ax.set_title(f'Joit {i%6+1}')
            ax.set_xlim(0, 100)  # 设置时间窗口的宽度
            ax.set_ylim(-2, 2)  # 设置 y 轴范围
            ax.grid(True)
            line_target, = ax.plot([], [], label='target_q', color='blue')
            line_q, = ax.plot([], [], label='q', color='red')
            lines_target.append(line_target)
            lines_q.append(line_q)
            ax.legend()

        # 数据缓冲区
        buffer_size = 100
        x_data = np.arange(buffer_size)
        y_target_data = np.zeros((12, buffer_size))
        y_q_data = np.zeros((12, buffer_size))

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path) # 从xml文件中加载模型
    # 获取base link的body id
    base_body_id = 1
    model.opt.timestep = cfg.sim_config.dt # 设置时间步长
    data = mujoco.MjData(model) # 初始化数据
    Default_Pos = np.zeros(12) # 初始化一个全零向量作为初始关节角度
    # 在仿真启动前应用初始关节角度
    initial_qpos = np.zeros_like(data.qpos)  # 初始化为全零
    init_joint_angles = { # 关节名称和初始角度，这里需要用自己的数据替换
       'left_hip_roll_joint': 0.,
        'left_hip_yaw_joint': 0.,
        'left_hip_pitch_joint': -14.884/180*3.14159,
        'left_knee_pitch_joint': 2*14.884/180*3.14159,
        'left_ankle_pitch_joint': -14.886/180*3.14159,
        'left_ankle_roll_joint': 0.,
        'right_hip_roll_joint': 0.,
        'right_hip_yaw_joint': 0.,
        'right_hip_pitch_joint': -14.884/180*3.14159,
        'right_knee_pitch_joint': 2*14.884/180*3.14159,
        'right_ankle_pitch_joint': -14.884/180*3.14159,
        'right_ankle_roll_joint': 0.,
    }

    # qpos 中的关节顺序与字典顺序一致
    joint_names = list(init_joint_angles.keys())

    for i, joint_name in enumerate(joint_names):
        joint_angle = init_joint_angles[joint_name]
        # 使用对应的索引 i 来设置 qpos 中的值
        data.qpos[i+7] = joint_angle
        Default_Pos[i] = joint_angle 
    
    
    # 设置基座的初始位置 ( pos[0:2] 代表 x, y, z 位置 , pos[3:6] 代表 pitch, yaw, roll 位置)
    data.qpos[0:3] = [0.0, 0.0, 0.93]  # 设置机器人在环境中的初始位置
    print(Default_Pos)

    mujoco.mj_step(model, data) # 
    viewer = mujoco_viewer.MujocoViewer(model, data) # 创建一个 Mujoco 视图

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double) # 初始化目标关节角度
    action = np.zeros((cfg.env.num_actions), dtype=np.double) # 初始化动作

    hist_obs = deque() # 创建一个队列用于存储历史观察值
    for _ in range(cfg.env.frame_stack): # 将初始关节角度添加到队列中
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 0
    time_to_stand_still = 0

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."): #  模拟时间60s/执行周期dt = 模拟次数

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        # 更新手柄数据
        joystick_controller.update_button_press()
        joystick_controller.update_cmd_vel()
        button_press = joystick_controller.get_button_press()
        cmd_vel = joystick_controller.get_cmd_vel()

        # 1000hz -> 100hz 因为Mujoco的dt是0.001s，而PPO的dt是0.01s，所以需要将Mujoco的dt缩小到0.01s
        if count_lowlevel % cfg.sim_config.decimation == 0:

            if hasattr(cfg.commands,"sw_switch"): # 用于重置步态相位，这样才能让机器人站立时没有步态信号，不会踏脚
                vel_norm = np.sqrt(cmd_vel["vx"]**2 + cmd_vel["vy"]**2 + cmd_vel["wz"]**2)
                if cfg.commands.sw_switch and vel_norm <= cfg.commands.stand_com_threshold:
                    time_to_stand_still+=1
                    if time_to_stand_still > 5:
                        count_lowlevel = 0
                else:
                    time_to_stand_still = 0

            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.85) # 这里的数字是步态周期，分别是左右腿的相位
            obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.85)
            obs[0, 2] = 1*(cmd_vel["vx"]) * cfg.normalization.obs_scales.lin_vel # 来自于手柄的指令
            obs[0, 3] = cmd_vel["vy"] * cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = cmd_vel["wz"] * cfg.normalization.obs_scales.ang_vel
            obs[0, 5:17] = (q - Default_Pos) * cfg.normalization.obs_scales.dof_pos #这里q要减去默认初始位置，因为训练的时候用的是当前状态和参考状态的差值。 qpos[5:17] 代表各个关节的角度值
            obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel # 关节角速度
            obs[0, 29:41] = action # 12维的动作数据，对应12个关节
            obs[0, 41:44] = omega
            obs[0, 44:47] = eu_ang

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations) # obs限幅

            hist_obs.append(obs) # 历史obs数据
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack): # 15帧的历史数据堆叠
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :] # 从历史obs数据里面取15帧obs作为模型输入
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy() # 输入policy_input到模型中，得到模型的输出作为action
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions) # action限幅

            target_q = action * cfg.control.action_scale + Default_Pos #模型的输出action是相对于default pos的相对目标位置，因为训练的时候q就是用的相对位置。所以要加上default pos变成绝对位置，便于PD控制器控制。
            # print(data.qvel[0:3])

            # 绘图
            if count_lowlevel % 100 == 0 and Plot_data:
            # 更新缓冲区
                y_target_data = np.roll(y_target_data, -1, axis=1)
                y_q_data = np.roll(y_q_data, -1, axis=1)
                y_target_data[:, -1] = target_q
                y_q_data[:, -1] = q

                # 更新绘图
                for i in range(12):
                    lines_target[i].set_data(x_data, y_target_data[i])
                    lines_q[i].set_data(x_data, y_q_data[i])
                    axes[i].relim()
                    axes[i].autoscale_view()
                
                plt.pause(0.001)  # 允许 matplotlib 刷新图形


        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps, # 使用PD控制对位置进行控制
                        target_dq, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau

        force = np.zeros(3)  # 外力初始化为0
    
        if button_press["A"]:
            force[0] = -FORCE_MAGNITUDE  # 向后施加推力
        elif button_press["Y"]:
            force[0] = FORCE_MAGNITUDE  # 向前施加推力

        # 如果有非零力，施加到base link
        if np.any(force):
            data.xfrc_applied[base_body_id][:3] = force

        mujoco.mj_step(model, data)
        viewer.render()

         # 清除外力
        data.xfrc_applied[base_body_id][:3] = np.zeros(3)
        count_lowlevel += 1

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()
    # env_cfg, _ = task_registry.get_cfgs(name=args.task)

    # class Sim2simCfg(XBotLCfg):

    #     class sim_config:
    #         if args.terrain:
    #             mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L-terrain.xml'
    #         else:
    #             mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L.xml'
    #         sim_duration = 60.0
    #         dt = 0.001
    #         decimation = 10

    #     class robot_config:
    #         kps = np.array([200, 200, 350, 350, 15, 15, 200, 200, 350, 350, 15, 15], dtype=np.double)
    #         kds = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.double)
    #         tau_limit = 200. * np.ones(12, dtype=np.double)

    class Sim2simCfg(D11Cfg):

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/Xbot/mjcf/XBot-L-terrain.xml'
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/d11/mjcf/D11_X_3ARM.xml'
            sim_duration = 120.0 # 模拟时间60s
            dt = 0.001
            decimation = 10

        class robot_config:
            kps = np.array([100, 100, 200, 200, 50, 25, 100, 100, 200, 200, 50, 25], dtype=np.double)
            kds = np.array([10, 10, 10, 10, 2, 1, 10, 10, 10, 10, 2, 1], dtype=np.double)
            tau_limit = 200. * np.ones(12, dtype=np.double)
    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
