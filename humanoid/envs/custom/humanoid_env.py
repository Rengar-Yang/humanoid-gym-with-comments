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

import os #导入库
import numpy as np #导入数学工具库并新命名为np

# from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from collections import deque

import torch
#from humanoid.envs import LeggedRobot
from humanoid.utils.terrain import HumanoidTerrain
from humanoid import LEGGED_GYM_ROOT_DIR
# humanoid.envs.base.base_task表示路径，BaseTask表示类
from humanoid.envs.base.base_task import BaseTask
# from humanoid.utils.terrain import Terrain
from humanoid.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from humanoid.utils.helpers import class_to_dict
from .humanoid_config import XBotLCfg

#获取三维的欧拉角
def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1) #维度上连接若干个张量。3*1
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi #？？？
    return euler_xyz


# from collections import deque

# 这个就相当于是环境文件

class XBotLFreeEnv(BaseTask):
    """
    XBotLFreeEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.  腿式机器人的配置文件
        sim_params: Parameters for the simulation.  模拟参数
        physics_engine: Physics engine used in the simulation.  模拟指定使用的物理引擎
        sim_device: Device used for the simulation.  模拟指定使用设备
        headless: Flag indicating whether the simulation should be run in headless mode.  指定模拟环境在运行时是否渲染

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        表示机器人上一个时间步的脚部在z轴上的位置，在许多机器人任务中，尤其是涉及到步态或平衡的任务，脚部的高度信息非常重要
        这个属性可以用来追踪和控制机器人的步态，

        feet_height (torch.Tensor): Tensor representing the height of the feet.
        表示脚部的高度，它可能包含了多个脚部的高度信息（如果机器人有多个脚部
        用于在强化学习中提供脚部的高度数据作为状态输入，对脚部高度的精确测量对平衡和稳定性控制至关重要

        sim (gymtorch.GymSim): The simulation object.
        这是一个仿真对象，用于与强化学习环境进行交互
        gymtorch是一个接口，用于机器人仿真，通过这个对象，可以与环境进行交互，进行动作执行，状态更新

        terrain (HumanoidTerrain): The terrain object.
        地形对象

        up_axis_idx (int): The index representing the up axis.
        表示上轴的索引。在三维空间中，通常有三个轴（x, y, z），up_axis_idx 指定了哪个轴被用作“上”方向
        确定机器人的“上”方向对平衡控制、重力计算和动作策略都很重要

        command_input (torch.Tensor): Tensor representing the command input.
        示命令输入。它通常包含了强化学习算法中用于控制机器人的动作的指令或控制信号
        作为强化学习算法的输入，用于生成控制命令以指导机器人在环境中的行为

        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        表示特权观察缓冲区。特权观察通常指额外的、对代理有用的状态信息，可能在训练时可用，但在实际应用时不可用
        在训练阶段使用这些特权观察来提供更丰富的环境信息，从而提升训练效果。实际使用时这些信息可能不可用或不被使用

        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        表示观察缓冲区。它包含了强化学习代理在每个时间步从环境中获取的状态信息
        用于记录和存储环境的观察数据，以供后续处理或分析。这是强化学习中的核心数据，用于训练和评估模型

        obs_history (collections.deque): Deque containing the history of observations.
        用于存储观察历史记录，在强化学习中，历史记录可以用于构建时序特征或状态
        例如，LSTM（长短期记忆网络）等方法可能需要利用观察历史来捕捉时序信息

        critic_history (collections.deque): Deque containing the history of critic observations.
        用于存储（critic）的历史观察
        在强化学习的 actor-critic 方法中，（critic）负责评估动作的价值。存储（critic）的历史数据可以帮助分析和改进价值函数的估计

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        这个方法用于在训练过程中向机器人施加随机外力，以模拟真实环境中的扰动。这有助于增强机器人对环境变化的适应性，提高策略的鲁棒性

        _get_phase(): Calculates the phase of the gait cycle.
        在机器人行走或奔跑的周期性任务中，相位表示机器人在步态周期中的位置。这对于协调机器人的动作，确保步态的平稳性和连续性至关重要

        _get_gait_phase(): Calculates the gait phase.
        步态相位用于标识机器人每个脚在步态周期中的状态（例如，是否接触地面）。这对于控制机器人的平衡和稳定性非常重要

        compute_ref_state(): Computes the reference state.
        参考状态通常是机器人理想的目标状态或动作，用于指导实际动作的执行
        通过参考状态，机器人可以更好地跟踪目标轨迹或姿态，提高动作的精度和效果

        create_sim(): Creates the simulation, terrain, and environments.
        这个方法用于初始化仿真环境，包括设置地形、物理参数和机器人本身。创建一个逼真的仿真环境对于训练有效的机器人控制策略至关重要

        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        在强化学习中，添加噪声可以模拟真实环境中的不确定性，提高策略的鲁棒性。这个方法根据配置设置噪声缩放向量，用于在观测值中添加噪声

        step(actions): Performs a simulation step with the given actions.
        这个方法在仿真中执行一个时间步，将动作应用到机器人上，并计算新的状态和观测值。这个过程是强化学习中策略评估和更新的核心部分

        compute_observations(): Computes the observations.
        这个方法整合和处理机器人的多种状态信息，生成用于策略和价值网络训练的观测值。观测值是强化学习算法的输入，对于训练有效的策略至关重要

        reset_idx(env_ids): Resets the environment for the specified environment IDs.
        在强化学习训练过程中，环境的重置用于清除状态和观测历史，重新开始新的训练轮次。这个方法确保每个环境在训练过程中能够被独立地重置和初始化

    """

    #*************************************初始化*********************************************

    def __init__(self, cfg: XBotLCfg, sim_params, physics_engine, sim_device, headless):
        """ 解析提供的配置文件，调用create_sim（）（用于创建、模拟、地形和环境），
        初始化训练期间使用的pytorch缓冲区

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
        #父函数的初始化
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        self.last_feet_z = 0.05  # 上一个时间步，机器人脚部在z轴上的位置
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)  # 脚部高度的大小就是（环境数目*2）
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))  # 清除观测历史和critic历史
        self.compute_observations()  # TODO


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

    def set_camera(self, position, lookat):
        """
        设置摄像头位置和方向
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, which will be called to compute the total reward.
            准备一系列奖励函数，这些函数将被调用以计算总奖励。
            寻找self._reward_<REWARD_NAME>，其中<reward_ NAME>是cfg中所有非零奖励量表的名称。
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # 准备功能列表
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            #以_reward_为开头的函数都是奖励函数
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # 奖励事件总数
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}


    # init buffers
    def _init_buffers(self):
        """
            初始化torch张量，其中包含模拟状态和处理量
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

        # 为不同的切片创建一些包装张量
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        # 初始化稍后使用的一些数据
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_rigid_state = torch.zeros_like(self.rigid_state)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # 关节位置偏移和PD增益
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[:, i] = 0.
                self.d_gains[:, i] = 0.
                print(f"PD gain of joint {name} were not defined, setting them to zero")


        self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.default_joint_pd_target = self.default_dof_pos.clone()
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(torch.zeros(
                self.num_envs, self.cfg.env.single_num_privileged_obs, dtype=torch.float, device=self.device))

    def _init_height_points(self):
        """ 返回高度测量值采样的点（在基架中）

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


    #*************************************创建仿真环境*********************************************

    def create_sim(self):
        """ 创建仿真、地形和环境
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _create_ground_plane(self):
        """ 将地平面添加到模拟中，根据cfg设置摩擦和恢复.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ 将高度场地形添加到模拟中，根据cfg设置参数。
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
        """ 将三角形网格地形添加到模拟中，根据cfg设置参数.
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


    def attach_camera(self, i, env_handle, actor_handle):
        """ 将相机添加到机器人身上.
        # """
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[0]
            camera_props.height = self.cfg.depth.original[1]
            camera_props.enable_tensors = True
            camera_horizontal_fov = self.cfg.depth.horizontal_fov
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)

            local_transform = gymapi.Transform()

            camera_position = np.copy(config.position)
            camera_angle = np.random.uniform(config.angle[0], config.angle[1])

            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)

            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)



    def _create_envs(self):
        """ 创建环境：
            1.加载机器人URDF/MJCF资产，
            2.对于每种环境
            2.1创造环境，
            2.2调用DOF和刚性形状属性回调，
            2.3使用这些属性创建actor并将其添加到env中
            3.存储机器人不同主体的索引
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

        # 从资源中保存正文名称
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        knee_names = [s for s in body_names if self.cfg.asset.knee_name in s]
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
        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)

        self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)

        for i in range(self.num_envs):
            # 创建env实例
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

            # %%%%%%%%%%%%%%%%%attach camera to robot%%%%%%%%%%%%%%%%%
            self.attach_camera(i, env_handle, robot_handle)

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])


    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ 回调允许存储/更改/随机化每个环境的刚性形状属性。
            在环境创建期间调用。
            基本行为：随机化每个环境的摩擦

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # 准备摩擦随机化
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 256
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

            self.env_frictions[env_id] = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ 回调允许存储/更改/随机化每个环境的DOF属性。
            在环境创建期间调用。
            基本行为：存储URDF中定义的位置、速度和扭矩限制

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
                self.dof_pos_limits[i, 0] = props["lower"][i].item() * self.cfg.safety.pos_limit
                self.dof_pos_limits[i, 1] = props["upper"][i].item() * self.cfg.safety.pos_limit
                self.dof_vel_limits[i] = props["velocity"][i].item() * self.cfg.safety.vel_limit
                self.torque_limits[i] = props["effort"][i].item() * self.cfg.safety.torque_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # 随机化基础质量
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        self.body_mass[env_id] = props[0].mass
        return props

    def _get_env_origins(self):
        """ 设置环境原点。在崎岖的地形上，原点由地形平台定义。
            否则，创建网格。
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            #将机器人放在地形定义的原点
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
            # 创建机器人网格
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.


    #************************************* step循环 *********************************************

    def step(self, actions):
        """
        在机器人仿真环境中执行动作，并应用一些动态随机化技术
        """
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action  # 就是将推荐动作加到动作上，这样机器人就能按照推荐动作进行运动，很简单的实现办法
        # dynamic randomization
        # 生成一个随机延迟向量 delay，其大小为 (self.num_envs, 1)，即每个环境一个随机延迟值。这个延迟值在 [0, 1) 范围内随机生成
        delay = torch.rand((self.num_envs, 1), device=self.device)
        # 将当前动作 actions 与之前的动作 self.actions 混合。混合的比例由随机延迟向量 delay 决定
        # 这样做可以引入随机性，使机器人的动作更加多样化，避免过拟合到特定的动作模式
        actions = (1 - delay) * actions + delay * self.actions

        # 在动作上添加随机噪声，噪声的强度由 dynamic_randomization 参数控制
        # 这个步骤引入了额外的随机性，以模拟现实世界中的不确定性和噪声，使学习到的策略更具有鲁棒性
        actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions) * actions
        #在子类中调用父类的方法
        #动作的限制幅度
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # 步进物理并渲染每一帧
        self.render() #render"可以表示将数据可视化成图表或图形的过程。
        #在这个循环中，“ _ ”是一个占位符，表示在循环体内不会使用到具体的迭代值，只关心循环的次数
        for _ in range(self.cfg.control.decimation):
            #view()相当于reshape、resize，重新调整Tensor的形状。
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            #假设 torques 是一个CUDA张量，torques处理后就是解包后的普通Tensor对象，可以在CPU上进行进一步的操作。
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            #sim哪里来的？
            self.gym.simulate(self.sim)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            #预计算
        self.post_physics_step()

        #对状态进行限制处理
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        # 返回剪切的obs、剪切的状态（无）、奖励、dones和info
        return self.obs_buf, self.privileged_obs_buf,  self.rew_buf, self.reset_buf, self.extras


    #************************************* 功能函数 *********************************************

    def _push_robots(self):
        """ 这个方法用于在训练过程中向机器人施加随机外力，以模拟真实环境中的扰动。
        """
        #定义最大的推力和力矩
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        #生成随机推力
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]
        #生成随机力矩
        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)
        self.root_states[:, 10:13] = self.rand_push_torque
        #写入gym
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _get_phase(self):
        # 步态周期是指机器人从一个特定姿态开始，经过一系列运动后回到同一姿态的时间
        cycle_time = self.cfg.rewards.cycle_time  # 机器人行走一个完整步态周期的理想时间

        # self.episode_length_buf是一个缓冲区，存储了当前episode的长度，即机器人已经经历的时间步数，每次仿真步都会更新这个值
        # dt是仿真时间步长
        # 计算当前仿真已经经历的总时间 / cycle_time 可以得到当前步态周期中的相位
        # 相位通常被规范化到 [0, 1) 的范围，表示在步态周期中的位置。例如，0 表示周期的开始，0.5 表示周期的中间，0.99 接近周期的结束
        # 但在这个函数中没有进行规范化处理
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        """
        用于计算机器人的步态相位掩码（gait phase mask）。步态相位掩码用于区分机器人在行走周期中的支撑相（stance phase）
        和摆动相（swing phase）,在机器人仿真和强化学习中，步态相位掩码有助于控制机器人的动作，使其更自然和稳定
        掩码的值为1表示支撑相（脚与地面接触），为0表示摆动相（脚在空中）
        这种掩码在机器人控制中非常有用，因为它可以帮助确定机器人的哪只脚在支撑地面，从而控制步态的稳定性和协调性
        """
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        #【0～2*pi】
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask

    def compute_ref_state(self):
        """
        计算机器人的参考状态，参考状态是一个目标状态或期望状态，它可以用来引导机器人的运动和学习
        """
        phase = self._get_phase()  # 获取步态周期的相位
        sin_pos = torch.sin(2 * torch.pi * phase)  # 计算步态周期的正弦值
        sin_pos_l = sin_pos.clone()  # 克隆用于左脚的相位值
        sin_pos_r = sin_pos.clone()  # 克隆用于右脚的相位值
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)  # 创建一个与关节位置相同尺寸的零张量，用于存储参考关节位置
        scale_1 = self.cfg.rewards.target_joint_pos_scale  # 获取目标关节位置的缩放比例
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        # 将左脚支撑相位的正值设为0，然后按比例调整关节位置。此过程将左脚在支撑相位时的关节位置设为默认位置
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 2+6] = sin_pos_l * scale_1
        self.ref_dof_pos[:, 3+6] = sin_pos_l * scale_2
        self.ref_dof_pos[:, 4+6] = sin_pos_l * scale_1
        # right foot stance phase set to default joint pos
        # 将右脚支撑相位的负值设为0，然后按比例调整关节位置。此过程将右脚在支撑相位时的关节位置设为默认位置
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 8+6] = sin_pos_r * scale_1
        self.ref_dof_pos[:, 9+6] = sin_pos_r * scale_2
        self.ref_dof_pos[:, 10+6] = sin_pos_r * scale_1
        # Double support phase
        # 当正弦值绝对值小于0.1时，将关节位置设为0，表示双脚都在地面支撑
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0
        # 将参考关节位置乘以2，得到参考动作
        self.ref_action = 2 * self.ref_dof_pos



    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: 5] = 0.  # commands
        noise_vec[5: 17+6] = noise_scales.dof_pos * self.obs_scales.dof_pos #关节位置
        noise_vec[17+6: 29+6*2] = noise_scales.dof_vel * self.obs_scales.dof_vel #关节速度
        noise_vec[29+6*2: 41+6*2] = 0.  # previous actions
        noise_vec[41+6*2: 44+6*2] = noise_scales.ang_vel * self.obs_scales.ang_vel  # 角速度
        noise_vec[44+6*2: 47+6*2] = noise_scales.quat * self.obs_scales.quat  # euler x,y
        return noise_vec

    def post_physics_step(self):
        """ 检查终止情况，计算观察结果和奖励
            calls self._post_physics_step_callback() 用于常见计算
            calls self._draw_debug_vis() （如果需要）
        """
        #更新机器人的状态
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities  获取机器人的位姿和速度
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        #获取手柄指令的航向并添加干扰
        self._post_physics_step_callback()

        # 计算观测值、奖励、重置
        self.check_termination() #检查是否需要重置
        self.compute_reward() #计算奖励
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        #在某些情况下，可能需要模拟步骤来刷新某些obs（例如身体位置）
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        #保存机器人上一次的动作状态
        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ 检查是否需要重置环境
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def compute_observations(self):
        """
        主要目的是计算和更新机器人的观测值（observations），这些观测值在强化学习中用于训练策略和价值网络
        函数使用了机器人仿真环境中的多种状态信息，并将其处理成适合用于强化学习训练的数据格式
        """
        # 获取相位信息，并计算双脚的参考状态
        phase = self._get_phase()
        self.compute_ref_state()

        # 计算相位的正弦和余弦值，并添加一个新的维度
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        # 获取步态相位和接触力掩码
        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.  #5牛用于判断是否落地

        # 生成命令输入，包括相位信息和缩放后的命令
        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        #q为相对于零位的变化量，dof_pos和dof_vel从哪里来？
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        diff = self.dof_pos - self.ref_dof_pos

        self.privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 3
            (self.dof_pos - self.default_joint_pd_target) * \
            self.obs_scales.dof_pos,  # 18 关节位置
            self.dof_vel * self.obs_scales.dof_vel,  # 18 关节速度
            self.actions,  # 18 动作
            diff,  # 18 关节微分
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3 base速度
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3 base姿态
            self.base_euler_xyz * self.obs_scales.quat,  # 3 base 姿态（四元素）
            self.rand_push_force[:, :2],  # 2 随机推力
            self.rand_push_torque,  # 3 随机推力力矩
            self.env_frictions,  # 1 摩擦
            self.body_mass / 30.,  # 1 质量
            stance_mask,  # 2 站立
            contact_mask,  # 2 接触
        ), dim=-1)

        obs_buf = torch.cat((
            self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
            q,  # 18D 关节角度
            dq,  # 18D 关节速度
            self.actions,  # 18D 关节动作
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3 bose速度
            self.base_euler_xyz * self.obs_scales.quat,  # 3 base姿态
        ), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                                 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        if self.add_noise:
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now) #观测历史
        self.critic_history.append(self.privileged_obs_buf) #评论历史

        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K
        #计算出obs和特权obs
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def compute_reward(self):
        """ Compute rewards
            调用每个非零刻度的奖励函数（在self_prepare_reward_function（）中处理）
            将每个术语添加到剧集总和和总奖励中
        """
        self.rew_buf[:] = 0.

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # 剪切后添加终止奖励
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew


    def _compute_torques(self, actions):
        """ 根据动作计算扭矩。
            动作可以被解释为给PD控制器的位置或速度目标，也可以直接解释为缩放扭矩。
            [注意]：扭矩必须与DOF的数量具有相同的尺寸，即使一些DOF没有被启动。
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        p_gains = self.p_gains
        d_gains = self.d_gains
        torques = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _draw_debug_vis(self):
        """ 绘制可视化以进行配音（大大降低了模拟速度）。
            默认行为：绘制高度测量点
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)


    def _get_heights(self, env_ids=None):
        """ 对每个机器人周围所需点的地形高度进行采样。
            这些点会因基座的位置而偏移，并因基座的偏航而旋转

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
        heightXBotL = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heightXBotL)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale



    #************************************* 重置 *********************************************

    def reset(self):
        """ 重置所有机器人"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False))

        return obs, privileged_obs


    def _post_physics_step_callback(self):
        """ 在计算终止、奖励和观察之前调用回调
            默认行为：根据目标和航向计算水平指令，计算测量的地形高度并随机推动机器人
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
        """ 随机选择某些环境的命令

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

    def _reset_dofs(self, env_ids):
        """ 重置选定环境的DOF位置和速度
            在0.5:1.5 x默认位置范围内随机选择位置。
            速度设置为零。

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ 重置选定环境的ROOT状态位置和速度
            根据课程设置基准位置
            选择-0.5:0.5[m/s，rad/s]范围内的随机基准速度
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
        # self.root_states[env_ids, 7:13] = torch_rand_float(-0.05, 0.05, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        if self.cfg.asset.fix_base_link:
            self.root_states[env_ids, 7:13] = 0
            self.root_states[env_ids, 2] += 1.8
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _update_terrain_curriculum(self, env_ids):
        """ 实施以游戏为灵感的课程。

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # 实施地形课程
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # 走得足够远的机器人会向更难的目标迈进
        move_up = distance > self.terrain.env_length / 2
        # 行走距离不到所需距离一半的机器人会前往更简单的地形
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # 解决最后一个级别的机器人被随机发送到一个级别
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """ 实施增加命令的课程

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def reset_idx(self, env_ids):
        """
        这段代码定义了一个名为 reset_idx 的函数，它继承并扩展了父类中的 reset_idx 方法
        该函数用于在强化学习训练中重置指定环境的状态，特别是清除观测历史和评论历史
        env_ids: 需要重置的环境索引。这通常是一个列表，包含多个需要重置的环境的索引
        """
        if len(env_ids) == 0:
            return
        # 更新课程
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # 避免在每一步都更新命令课程，因为最大命令对所有env都是通用的
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        # 重置机器人状态
        self._reset_dofs(env_ids)

        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # 重置缓冲区
        self.last_last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_rigid_state[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # 填充额外内容
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # 记录其他课程信息
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # 向算法发送超时信息
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # 修复重置重力错误
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])

        # 清除观测历史和评论历史
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0



    # ================================================ Rewards ================================================== #


    def _reward_joint_pos(self):
        """
        根据当前关节位置和目标关节位置之间的差值计算奖励。
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_feet_distance(self):
        """
        根据双脚之间的距离计算奖励。惩罚脚彼此靠近或太远。
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self):
        """
        根据人形动物膝盖之间的距离计算奖励。
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_foot_slip(self):
        """
        计算减少脚滑的奖励。奖励基于接触力
        以及脚的速度。接触阈值用于确定脚是否接触
        与地面。脚的速度由接触条件计算和缩放。
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_feet_air_time(self):
        """
        计算脚部腾空时间的奖励，促进更长的步数。这是通过以下方式实现的
        检查在空中后与地面的第一次接触。播出时间为
        限制为奖励计算的最大值。
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        """
        根据与步态阶段对齐的脚接触次数计算奖励。
        奖励或惩罚取决于脚接触是否与预期的步态阶段相匹配。
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        """
        计算保持base方向的奖励。它惩罚偏差
        使用base欧拉角和投影重力矢量从所需的base方向开始。
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_feet_contact_forces(self):
        """
        计算将接触力保持在指定范围内的奖励。惩罚
        脚上的高接触力。
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :],
                                     dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

    def _reward_default_joint_pos(self):
        """
        计算保持联合头寸接近默认头寸的奖励，并关注
        关于惩罚偏航和横滚方向的偏差。主罚不包括偏航和横滚。
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = joint_diff[:, 6:8]  # joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 12: 14]  # joint_diff[:, 6: 8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_base_height(self):
        """
        根据机器人的基础高度计算奖励。惩罚偏离目标基准高度的行为。
        奖励是根据机器人底座和平均高度之间的高度差计算的
        当脚与地面接触时。
        """
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self):
        """
        根据基地的加速度计算奖励。惩罚机器人基座的高加速度，
        鼓励更平稳的运动。
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def _reward_vel_mismatch_exp(self):
        """
        根据机器人的线速度和角速度的不匹配计算奖励。
        通过惩罚较大的偏差来鼓励机器人保持稳定的速度。
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        计算准确跟踪线速度和角速度命令的奖励。
        惩罚与指定线速度和角速度目标的偏差。
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    def _reward_tracking_lin_vel(self):
        """
        沿xy轴跟踪线速度命令。
        根据机器人的线速度与指令值的匹配程度计算奖励。
        """
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        跟踪偏航旋转的角速度命令。
        根据机器人的角速度与指令偏航值的匹配程度计算奖励。
        """

        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_feet_clearance(self):
        """
        根据运动过程中摆动腿与地面的间隙计算奖励。
        鼓励在步态摆动阶段适当抬起脚。
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        """
        根据机器人相对于指令速度的速度来奖励或惩罚机器人。
        此功能检查机器人是否移动得太慢、太快或以期望的速度移动，
        并且如果移动方向与命令匹配。
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)

    def _reward_torques(self):
        """
        惩罚机器人关节中使用高扭矩的行为。通过最小化来鼓励高效移动
        发动机施加的必要力。
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        在机器人的自由度（DOF）处惩罚高速。这鼓励更平稳和
        更可控的动作。
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        """
        惩罚机器人自由度（DOF）的高加速度。这对于确保
        运动平稳，减少机器人机械部件的磨损。
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        """
        惩罚机器人与环境的碰撞，特别是关注选定的身体部位。
        这鼓励机器人避免与物体或表面发生不必要的接触。
        """
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_action_smoothness(self):
        """
        通过惩罚连续动作之间的巨大差异来鼓励机器人动作的流畅性。
        这对于实现流体运动和减少机械应力非常重要。
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3
