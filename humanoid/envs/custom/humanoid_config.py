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

import numpy as np
from humanoid.envs.base.base_config import BaseConfig

class XBotLCfg(BaseConfig):
    """
    人形机器人环境的配置文件.
    """
    class env:
        """
        env类主要配置机器人仿真环境的各个方面，观察空间，动作空间，环境参数等
        """
        # change the observation dim
        # 指定了在观察中使用的连续帧的数量，使用帧堆栈可以帮助agent捕捉时间上的连续性和运动信息，单独帧可能不足以提供足够的信息来做出决策
        frame_stack = 15
        # 特权观察的帧堆栈的大小，提供额外的时序信息
        c_frame_stack = 3
        # 动作空间维度
        num_actions = 18
        # 每一帧 观察空间的维度数目，包括机器人的位置、速度、角速度、动作空间、传感器等
        num_single_obs = 47 + 6*3
        # 所以整个观察空间的维度数量就是（单个维度*帧数）
        num_observations = int(frame_stack * num_single_obs)
        # 每一帧 特权观察空间的维度数目
        single_num_privileged_obs = 73 + 6*4
        # 所以整个特权观察空间的维度数量就是（单个维度*帧数）
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        # not used with heightfields/trimeshes
        env_spacing = 3.
        # send time out information to the algorithm
        send_timeouts = True
        # 环境的实例数量，即一次仿真中有4096个环境实例，但是只需要开一个物理引擎
        num_envs = 4096
        # 定义了每个episode的长度，单位是秒，一个episode是agent从开始到结束的一个完整的交互周期
        # 相当于是对episode的时间长度进行了限制，遇到需要快速反应和调整的任务，较短的episode长度可能更合适
        episode_length_s = 24     # episode length in seconds
        # 指示是否使用参考动作，参考动作是机器人的期望动作，而不是实际动作，参考动作可以减少训练时间，但是可能会影响训练效果
        use_ref_actions = False   # speed up training by using reference actions
        #历史信息的长度
        num_observation_history_len = 1

    class safety:
        """
        机器人在仿真中应遵循的安全限制
        """
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset:
        """
        用于定义和配置机器人模型的资产
        """
        # 配置机器人的URDF文件路径，URDF 文件描述了机器人的物理结构，包括其关节、链接、传感器等
        # 在仿真环境中，URDF 文件用于加载和定义机器人的模型和物理特性，以便在仿真中进行准确的操作和测试
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/D11_X/urdf/D11_X_3ARM.urdf'
        # 机器人模型名称
        name = "D11_X"
        # 机器人脚部关节名称
        foot_name = "ankle_roll"
        # 机器人膝关节名称
        knee_name = "knee"
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)

        # 指定在发生接触时需要终止仿真的部件名称，这个设置用于防止机器人在仿真过程中发生不安全或不合适的接触
        terminate_after_contacts_on = ['base_link']
        # 在强化学习任务中，接触指定部件（如 "base_link"）会导致奖励减少，从而引导代理学习到避免这些接触的策略
        # 惩罚接触有助于训练出更稳定和安全的控制策略
        penalize_contacts_on = ["base_link"]
        # 用于控制是否启用机器人自碰撞检测。0 表示启用自碰撞检测，1 表示禁用
        # 启用自碰撞检测可以识别机器人部件之间的碰撞，从而提供更准确的仿真和控制数据
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        # 决定是否翻转机器人模型的视觉附件  TODO 不太理解这个用来干什么
        flip_visual_attachments = False
        # 决定是否将机器人模型中的圆柱体形状替换为胶囊体
        # 在某些仿真中，使用胶囊体替代圆柱体可以更准确地模拟碰撞行为，尤其是对于有圆形或弯曲部件的机器人
        replace_cylinder_with_capsule = False
        # 决定是否固定机器人的基座链接
        # 如果基座链接被固定，机器人在仿真中将保持不动，这可以用于测试其他部件的行为或在固定的环境中进行分析
        fix_base_link = False

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class terrain:
        """
        用于定义和配置环境中的地形特性
        """
        # mesh_type指定地形的网格类型
        # plane表示平面网格，用于模拟简单的平坦地面,trimesh表示三角网格，用于模拟更复杂的地形，能够表示不规则的地形特征
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        # 决定是否使用渐进式训练，即逐渐增加环境的复杂度
        curriculum = False
        # rough terrain only:
        # rough terrain only:
        # 决定是否测量地形的高度，平坦地面是不需要测量地面高度的，复杂的地形需要
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        # 静态摩擦系数和动态摩擦系数，一个描述物体在静止状态下的摩擦力，一个描述物体在运动状态下的摩擦力
        static_friction = 0.6
        dynamic_friction = 0.6
        # 地形的长度和宽度
        terrain_length = 8.
        terrain_width = 8.
        # 地形网格的行数和列数，用于定义地形的网格分布，较高的行列数可以提供更详细的地形特征
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # 地形初始化等级，用于定义地形的初始状态，数值越高，地形越复杂， 需要与curriculum系数一起使用
        max_init_terrain_level = 10  # starting curriculum state
        # terrain_proportions是一个列表，指定不同地形类型的比例，平面，障碍物，均匀地面，上坡，下坡，楼梯
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        # 地形的弹性系数，也称为恢复系数，决定地形与物体碰撞之后的弹性程度，设置为0，表示没有弹回的效果
        restitution = 0.
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces


    class noise:
        """
        在机器人仿真和强化学习中，噪声通常用于模拟现实世界中的不确定性、传感器误差或环境扰动
        添加噪声可以帮助训练代理应对不确定性，从而提高其鲁棒性和适应能力
        """
        # 决定是否在仿真中添加噪声
        add_noise = True
        # 缩放因子，用于调整噪声的强度，合适的噪声水平有助于使仿真更接近实际应用场景
        noise_level = 0.6    # scales other values

        class noise_scales:
            """
            定义了不同类型的噪声及其对应的强度
            """
            dof_pos = 0.05 # 关节位置的噪声缩放因子
            dof_vel = 0.5 # 关节速度的噪声缩放因子
            ang_vel = 0.1 # 角速度的噪声缩放因子
            lin_vel = 0.05 # 线速度的噪声缩放因子
            quat = 0.03 # 四元数的噪声缩放因子，四元数可以有效地处理机器人在三维空间中的旋转，模拟真实环境中的运动
            height_measurements = 0.1 # 高度测量的噪声缩放因子
            gravity = 0.05

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class init_state:
        """
        定义机器人的初始状态，包括机器人的位置，关节角度等
        """
        pos = [0.0, 0.0, 0.99]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # 定义了机器人的各个关节在仿真开始时的初始角度
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "arm_left_shoulder_pitch_joint": 0.0,
            "arm_left_shoulder_roll_joint": 0.0,
            "arm_left_elbow_pitch_joint": 1.0472,
            "arm_right_shoulder_pitch_joint": 0.0,
            "arm_right_shoulder_roll_joint": 0.0,
            "arm_right_elbow_pitch_joint": 1.0472,
            "leg_left_hip_roll_joint": 0.0,
            "leg_left_hip_yaw_joint": 0.0,
            "leg_left_hip_pitch_joint": -0.1580,
            "leg_left_knee_pitch_joint": 0.3966,
            "leg_left_ankle_pitch_joint": -0.2386,
            "leg_left_ankle_roll_joint": 0.0,
            "leg_right_hip_roll_joint": 0.0,
            "leg_right_hip_yaw_joint": 0.0,
            "leg_right_hip_pitch_joint": -0.1580,
            "leg_right_knee_pitch_joint": 0.3966,
            "leg_right_ankle_pitch_joint": -0.2386,
            "leg_right_ankle_roll_joint": 0.0,
        }

    class control:
        # PD Drive parameters:
        """
        用于设置和管理机器人的控制参数。这个类的设置对于机器人在仿真中的动态行为和稳定性至关重要
        """
        # Proportional-Derivative Control是控制系统中常用的一种类型，结合了比例（P）和微分（D）控制，以确保系统的稳定性和响应性
        # PD Drive parameters:
        # 刚度：用于设置各个关节的比例增益（P增益）。它控制了关节对偏离目标角度的反应程度
        # 高刚度值使得控制器对角度误差的反应更强，快速纠正偏差；低刚度值则会导致反应较慢
        # roll:
        # pitch:
        # yaw:
        stiffness = {
            "shoulder_pitch": 75,
            "shoulder_roll": 75,
            "elbow_pitch": 10,
            "hip_roll": 150,
            "hip_yaw": 150,
            "hip_pitch": 200,
            "knee_pitch": 200,
            "ankle_pitch": 20,
            "ankle_roll": 10
        }
        # 阻尼：用于设置各个关节的微分增益（D增益）。它控制了关节对速度变化的反应，以减少震荡和过度反应
        # 高阻尼值可以减少系统的震荡和过冲，提高系统的稳定性。低阻尼值可能导致系统反应过度并产生震荡
        damping = {
            "shoulder_pitch": 3,
            "shoulder_roll": 3,
            "elbow_pitch": 1,
            "hip_roll": 6,
            "hip_yaw": 6,
            "hip_pitch": 8,
            "knee_pitch": 8,
            "ankle_pitch": 2,
            "ankle_roll": 1
        }
        # 定义了控制输入（动作）的缩放因子，action是控制输入，defaultAngle是默认角度
        # 用于将控制信号（动作）缩放到实际的角度值，可以控制输入的范围，适应不同的任务需求
        # 将强化学习算法产生的控制信号转换为适合机器人控制系统的实际角度值
        # action scale: target angle = actionScale * action + defaultAngle
        # action_scale = np.array([0.1, 0.1, 0.1,
        #                          0.1, 0.1, 0.1, # arm
        #                          0.25, 0.25, 0.25, 0.25, 0.05, 0.05,
        #                          0.25, 0.25, 0.25, 0.25, 0.05, 0.05])
        action_scale = 0.25
        # 控制动作更新的采样率，定义了每个仿真时间步中应用控制动作的次数
        # 为 10 意味着每 10 个仿真时间步应用一次控制动作，因此控制频率为 100 Hz（如果仿真时间步为 0.01 秒）
        # decimation: Number of control action updates @ sim DT per policy DT
        # 增加 decimation 值可以减少控制频率，降低计算负担，但可能影响控制的平滑性。减小 decimation 值可以提高控制频率，增加控制的响应性
        # 仿真系统每0.001s（下面sim参数里定义了仿真时间步）进行一次状态更新，10个仿真时间步之后，才会执行一次动作
        # 因此，控制的动作频率实际上是100hz，每隔0.01s仿真执行一次动作
        decimation = 10  # 100hz

    class sim:
        """
        仿真环境参数设置，尤其是物理引擎PhysX相关的配置
        """
        # 仿真时间步，表示每个仿真时间步长（以秒为单位）表示仿真每秒进行 1000 次更新
        # 仿真的时间分辨率非常高。较小的时间步可以提供更精确的仿真结果，但也会增加计算负担
        dt = 0.001  # 1000 Hz
        # 仿真的子步数，表示每个仿真时间步长被拆分成多少个子步，每个子步进行物理模拟 TODO
        # 增加substeps的值可以提高仿真的稳定性和准确性，但也会增加计算量。将其设置为2，表示在每个时间步内执行两个子步，这可以改善仿真的准确性
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        # 确定仿真中的上方向轴 设置为 1 表示 z 轴是上方向轴。这是对仿真世界的垂直方向的定义，在某些仿真环境中，默认的上方向可能是 y 轴
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            """
            这些参数控制了使用 NVIDIA PhysX 引擎进行物理仿真的设置
            """
            # 物理引擎使用的线程数量,多线程可以加速物理计算，提高仿真效率。根据计算资源，选择适当的线程数可以平衡性能和计算开销
            num_threads = 10
            # 物理引擎的求解器类型 0 表示 PGS（Projected Gauss-Seidel），1 表示 TGS（TGS）
            # TGS 通常比 PGS 更适合处理复杂的动力学问题，提供更好的稳定性和精度
            solver_type = 1  # 0: pgs, 1: tgs
            # 求解位置约束的迭代次数,位置迭代次数越多，约束解算的精度越高，但计算负担也会增加
            num_position_iterations = 4
            # 求解速度约束的迭代次数,设置为 0 表示不进行速度约束的额外迭代，通常速度约束的精度可以通过其他参数控制
            num_velocity_iterations = 1
            # 碰撞检测的偏移距离,用于避免物体之间的穿透，设置为 0.01 米可以确保碰撞检测的稳定性
            contact_offset = 0.01  # [m]
            # 物体恢复偏移量, 确保物体在碰撞后的恢复位置。设置为 0.0 表示没有恢复偏移量
            rest_offset = 0.0   # [m]
            # 碰撞反弹的阈值速度,低于该速度的碰撞不会导致反弹，这有助于减少高频震荡或不自然的碰撞反应
            bounce_threshold_velocity = 0.1  # [m/s]
            #  最大的穿透修正速度,如果物体在碰撞后穿透了其他物体，最大修正速度为 1.0 米/秒，用于控制穿透修正的最大速度
            max_depenetration_velocity = 1.0
            # GPU上处理的最大碰撞对数,用于控制在 GPU 上处理的最大碰撞对数，以适应更大的仿真环境或更多的数据
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            # 默认缓冲区大小的乘数,用于调整物理引擎内部缓冲区的大小，以适应更大的仿真环境或更多的数据
            default_buffer_size_multiplier = 5
            # 碰撞数据的收集方式, 0 表示从不收集
            # 1 表示在最后一个子步骤收集，2 表示在所有子步骤中收集。设置为 2 表示在每个子步骤都收集碰撞数据，这有助于提高碰撞检测的精度
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        """
        通过在训练过程中随机化仿真环境中的某些参数，可以使模型在更广泛的真实世界情景中表现得更好
        """
        # 随机化摩擦系数，可以提高模型的泛化能力，但需要谨慎使用，因为随机化的摩擦系数可能会影响模型的性能
        randomize_friction = True
        # 随机化摩擦系数的范围，可以控制随机化的摩擦系数的大小，以适应不同的任务需求
        friction_range = [0.1, 2.0]
        # 随机化机器人的基础质量，可以提高模型的泛化能力，但需要谨慎使用，因为随机化的质量可能会影响模型的性能
        randomize_base_mass = True
        # 随机化机器人的基础质量范围，可以控制随机化的质量大小，以适应不同的任务需求
        added_mass_range = [-5., 5.]
        # 随机化机器人的推力，可以提高模型的泛化能力
        # 仿真过程中会随机对机器人施加外力，以模拟碰撞或其他外部干扰。这有助于训练机器人在受到干扰时保持平衡和恢复
        push_robots = True
        # 随机化机器人的推力间隔，表示机器人每隔4s对机器人施加一次外力，模拟机器人在真实环境中遇到的周期性干扰
        push_interval_s = 4
        # 随机化机器人的推力速度范围，表示机器人对机器人施加的外力速度范围，这个参数控制了外力的强度
        max_push_vel_xy = 0.2
        # 随机化机器人的推力角速度范围，表示机器人对机器人施加的外力角速度范围，这个参数控制了外力引起的旋转强度
        max_push_ang_vel = 0.4
        # dynamic randomization
        dynamic_randomization = 0.02

    class commands:
        """
        命令（commands）用于控制机器人执行特定的动作和行为
        """
        curriculum = False
        max_curriculum = 1.
        # lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        # 这里定义了四个不同的命令，这些命令控制机器人的线速度（x 方向和 y 方向）、角速度（绕 z 轴旋转）和航向
        num_commands = 4
        # 命令重新采样的时间间隔（秒）,每隔 8 秒重新生成一组新的命令
        # 这意味着机器人在仿真中每 8 秒会接收到新的控制指令，以保持动作和行为的变化和适应性
        resampling_time = 8.  # time before command are changed[s]
        # 是否根据航向误差计算角速度命令
        # 机器人将根据当前的航向误差计算角速度命令。这有助于机器人保持或调整其行进方向
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            # 机器人的 x 方向线速度将在 -0.3 到 0.6 米/秒之间随机变化
            # 这意味着机器人可以向前和向后移动，并且向前的最大速度为 0.6 米/秒，向后的最大速度为 0.3 米/秒
            lin_vel_x = [-0.3, 0.6]  # min max [m/s]
            # 机器人的 y 方向线速度将在 -0.3 到 0.3 米/秒之间随机变化
            # 这意味着机器人可以向左和向右移动，最大速度均为 0.3 米/秒
            lin_vel_y = [-0.3, 0.3]  # min max [m/s]
            # 机器人的角速度将在 -0.3 到 0.3 弧度/秒之间随机变化
            # 这意味着机器人可以顺时针和逆时针旋转，最大角速度均为 0.3 弧度/秒，绕z轴旋转
            ang_vel_yaw = [-0.3, 0.3]  # min max [rad/s]
            # 机器人的航向在 -3.14 到 3.14 弧度（-π 到 π 弧度）之间变化，这表示机器人可以调整其航向到任何方向
            # 机器人的航向（Heading）是指机器人在水平面上所朝的方向
            # 航向定义了机器人在二维平面（通常是 x-y 平面）中的朝向角度，它是机器人导航和控制中的一个关键概念
            heading = [-3.14, 3.14]

    class rewards:
        """
        奖励函数（class rewards）是关键组成部分，它用于指导强化学习算法的训练过程
        奖励函数通过对机器人行为的评价来决定哪些动作是好的，哪些是需要改进的
        """

        base_height_target = 0.94
        #双脚之间的距离，最小和最大
        min_dist = 0.2
        max_dist = 0.5
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17  # rad
        target_feet_height = 0.1  # m
        # 机器人行走周期的理想时间。如果实际周期接近这个时间，则会获得奖励
        cycle_time = 0.64  # sec  步态频率
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        # 是否只使用正奖励,如果设置为 True，则负奖励会被裁剪为零。这可以避免在训练初期由于负奖励导致的提前终止问题
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        # 最大接触力, 超过这个值的接触力会被罚分，鼓励机器人保持在合理的力范围内以避免损坏或不稳定
        max_contact_force = 700  # Forces above this value are penalized

        class scales:
            # reference motion tracking
            joint_pos = 1.6
            feet_clearance = 1.
            feet_contact_number = 1.2
            # gait
            feet_air_time = 1.
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            # lin_vel_z = -2.0
            # ang_vel_xy = -0.05
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 0.5
            orientation = 1.
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.
            #
            termination = -0.0
            feet_stumble = -0.0
            action_rate = -0.
            stand_still = -0.

    class normalization:
        """
        数据的标准和归一化是非常重要的步骤，确保不同尺度的数据在算法处理中具有相似的影响力
        """

        class obs_scales:
            """
            定义了不同类型观测值的缩放因子，这些因子用于标准化不同类型的观测数据
            """
            # 线速度的缩放因子，将线速度观测值乘以 2，以调整其在训练过程中的影响力
            lin_vel = 2.
            # 角速度的缩放因子，不做缩放
            ang_vel = 1.
            # 关节角度的缩放因子，不做缩放
            dof_pos = 1.
            # 关节速度的缩放因子，将关节速度观测值乘以 0.05，以调整其在训练过程中的影响力
            dof_vel = 0.05
            # 四元数的缩放因子，保持原值
            quat = 1.
            # 高度测量值的缩放因子，将高度测量值乘以 5，以调整其在训练过程中的影响力
            height_measurements = 5.0

        # 观测值的截断范围，将观测值限制在 -18 到 18 之间
        # 以避免极端值对训练过程产生不利影响。这种截断可以防止异常值（outliers）影响模型的学习
        clip_observations = 18.
        # 动作值的截断范围，将动作值限制在 -18 到 18 之间，以确保生成的动作在合理范围内
        # 这可以防止动作过大或过小导致的控制问题或仿真不稳定
        clip_actions = 18.


class XBotLCfgPPO(BaseConfig):
    seed = 5
    runner_class_name = 'OnPolicyRunner'  # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0  # 初始化噪声标准差
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.001  # 熵系数，用于控制策略的熵正则化项，增加策略的随机性，从而促进探索。熵系数越高，策略越随机
        learning_rate = 1e-5
        schedule = 'adaptive' # could be adaptive, fixed
        num_learning_epochs = 2  # 每次策略更新时的学习轮数
        gamma = 0.994  # 折扣因子，用于计算未来奖励的折扣值，以平衡长期 vs 短期奖励，越大则未来回报的影响越大
        lam = 0.9  # GAE系数，用于平衡偏差和方差。GAE用于计算优势函数，结合了TD（Temporal Difference）误差和蒙特卡洛（Monte Carlo）估计
        num_mini_batches = 4  # 每次策略更新中将数据划分为的小批量数量。增加小批量数量可以减少每次更新的方差
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'

        # 每个环境在每次策略更新之前所采样的时间步数
        # 每次策略更新之前，智能体在环境中运行并收集数据。这些数据包括状态、动作、奖励和下一状态等信息
        # PPO是on-policy的算法，意味着每次策略更新只使用当前策略生成的数据
        num_steps_per_env = 60  # per iteration

        max_iterations = 3001  # number of policy updates

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'XBot_ppo'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
