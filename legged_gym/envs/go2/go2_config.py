from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        hip_reduction = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class commands( LeggedRobotCfg.commands ):
        # 前 1500 次训练迭代将完全静止命令的采样概率从 0 线性增加到 10%。
        zero_command_curriculum = {'start_iter': 0, 'end_iter': 1500, 'start_value': 0.0, 'end_value': 0.1}

    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/TOE_dog4.5/urdf/dog.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.32
        # 前 1500 次训练迭代将竖直速度惩罚从 -2.0 线性放松到 -0.5。
        curriculum_rewards = [
            {"reward_name": "lin_vel_z", "start_iter": 0, "end_iter": 1500, "start_value": 1.0, "end_value": 0.75},
            {'reward_name': 'base_height', 'start_iter': 0, 'end_iter': 5000, 'start_value': 1.0, 'end_value': 10.0},
        ]
        # Go2 对角腿为同相足：左前-右后同步，右前-左后同步，两组交替运动。
        gait_synced_feet = (
            ("FL_foot", "RR_foot"),
            ("FR_foot", "RL_foot"),
        )
        gait_sigma = 0.05
        gait_max_error = 0.2
        gait_velocity_threshold = 0.5
        # 分别比较两组对角腿相对默认站姿的关节偏移。
        joint_mirror_joint_groups = (
            (
                ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"),
                ("RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"),
            ),
            (
                ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"),
                ("RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"),
            ),
        )
        # 髋外展关节镜像时反号，大腿和小腿关节同号。
        joint_mirror_signs = (-1.0, 1.0, 1.0)
        # 足端顺序沿用 feet_names；0/0.5 将两组对角腿错开半个周期。
        phase_foot_trajectory_cycle_time = 0.8
        phase_foot_trajectory_phase_offsets = (0.0, 0.5, 0.5, 0.0)
        phase_foot_trajectory_stance_ratio = 0.5
        # 当前前后摆幅为 0，只约束摆动相的竖直抬脚高度。
        phase_foot_trajectory_horizontal_span = 0.15
        phase_foot_trajectory_swing_height = 0.08
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -1e-4
            dof_pos_limits = -1.
            dof_vel_limits = -2.
            # 正权重项：鼓励对角步态和相位轨迹跟踪。
            feet_gait =  0 # 0.1
            phase_foot_trajectory_exp = 0 # 0.2
            # 负权重项：惩罚关节不对称、支撑脚打滑和过大的落脚速度。
            joint_mirror = -0.05
            feet_slide = -0.05
            foot_impact_velocity = -0.0
            # 该项使用正权重，零指令时按落地脚数量奖励稳定站立。
            feet_contact_without_cmd = 0.05 #0.1

class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        # experiment_name = 'rough_go2_tshim'
        experiment_name = 'dog4.5_cts'

  
