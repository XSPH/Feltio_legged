# CTS-HIM 融合算法手动修改方案

## 1. 文档目标

本文给出一条可以逐步手动实现、每一步都能够单独检查的 CTS-HIM 融合路线。目标仓库是当前的 Feltio_legged，两个外部仓库仅作为参考代码来源：

- CTS 参考：/home/asuka/Legged/go2_rl_gym
- HIM 参考：/home/asuka/Legged/HIMLoco
- 目标仓库：/home/asuka/Legged/Feltio_legged

最终算法采用“CTS 作为训练框架，HIM 作为学生状态表征”的组合：

- CTS 负责划分教师和学生环境，并用两组轨迹共同更新共享策略。
- HIM 负责从历史本体感知中估计速度和隐式机器人响应。
- 教师特权表征只作为学生表征的辅助约束，不能取代学生自己的 PPO 轨迹。

本文只规划修改，不要求一次完成全部代码。建议严格按照阶段顺序推进，每个阶段通过验收后再进入下一阶段。

## 2. 先固定算法定义

### 2.1 教师分支

教师在仿真中访问特权观测 p_t：

~~~text
p_t -> teacher_encoder -> z_teacher
~~~

教师上下文定义为：

~~~text
teacher_context = concat(true_base_velocity, z_teacher)
~~~

建议使用以下维度：

- true_base_velocity：3
- z_teacher：16
- teacher_context：19

教师动作由共享 Actor 产生：

~~~text
action_teacher = actor(concat(current_obs, teacher_context))
~~~

### 2.2 学生分支

学生只能访问历史本体感知：

~~~text
obs_history -> him_source_encoder -> predicted_velocity, z_him
~~~

学生上下文定义为：

~~~text
student_context = concat(predicted_velocity, z_him)
~~~

维度与教师保持一致：

- predicted_velocity：3
- z_him：16
- student_context：19

学生动作同样由共享 Actor 产生：

~~~text
action_student = actor(concat(current_obs, student_context))
~~~

### 2.3 共享模块

教师和学生共享：

- Actor；
- Critic；
- 动作标准差参数；
- PPO 优化目标。

不共享：

- teacher_encoder；
- him_source_encoder；
- him_target_encoder；
- HIM prototypes。

建议 Critic 按 CTS 的设计接收：

~~~text
critic_input = concat(privileged_obs, context.detach())
~~~

如果调试阶段怀疑 Critic 输入导致问题，可以临时退化为只输入 privileged_obs，但正式对比实验中需要明确记录这一变化。

## 3. 当前仓库的固定接口

在修改算法前，先把当前训练接口当作不可随意改变的契约。

### 3.1 当前一步 Actor 观测

当前 legged_gym/envs/base/legged_robot.py 中 Actor 观测为 45 维：

| 索引 | 内容 | 维度 |
| --- | --- | ---: |
| 0:3 | base angular velocity | 3 |
| 3:6 | projected gravity | 3 |
| 6:9 | velocity commands | 3 |
| 9:21 | joint position offsets | 12 |
| 21:33 | joint velocities | 12 |
| 33:45 | previous actions | 12 |

不要为了照搬 HIM 源码而调整此顺序。HIM 参考仓库使用了不同的观测排列，直接复制切片会产生静默错误。

### 3.2 当前特权观测

当前 privileged_obs 为 247 维：

| 索引 | 内容 | 维度 |
| --- | --- | ---: |
| 0:45 | 无噪声的一步本体观测 | 45 |
| 45:48 | scaled base linear velocity | 3 |
| 48:60 | normalized foot contact forces | 12 |
| 60:247 | terrain height measurements | 187 |

因此最终建议的关键维度是：

| 张量 | 维度 |
| --- | ---: |
| current_obs | 45 |
| privileged_obs | 247 |
| history_length | 5 |
| flattened_history | 225 |
| teacher/student context | 19 |
| Actor 输入 | 64 |
| Critic 输入 | 266 |
| action | 12 |

第一版统一使用 5 帧历史，排列约定为“最旧帧在前，当前帧在最后”：

~~~text
[o_(t-4), o_(t-3), o_(t-2), o_(t-1), o_t]
~~~

HIM 论文和参考代码对 H 的计数方式不完全一致。第一版不要同时尝试 5 帧和 6 帧，等融合算法稳定后再做历史长度消融。

### 3.3 HIM 的下一时刻响应目标

HIM target encoder 不应该编码下一时刻的 command，因为 command 不是机器人对环境的响应。按照当前观测顺序构造 45 维 target_response：

~~~python
next_response = torch.cat(
    [
        next_privileged_obs[:, 0:6],
        next_privileged_obs[:, 9:45],
        next_privileged_obs[:, 45:48],
    ],
    dim=-1,
)
~~~

对应内容为：

- angular velocity：3；
- projected gravity：3；
- joint position、joint velocity、previous action：36；
- base linear velocity：3。

总维度为 45。

速度监督建议使用当前时刻真实速度：

~~~python
velocity_target = privileged_obs[:, 45:48]
~~~

这样学生在执行时刻 t 估计的是当前速度。HIM 参考实现使用下一时刻特权观测提取速度，若后续希望严格复现源实现，可以将“当前速度”和“下一时刻速度”作为单独消融，不要在首次融合时混用。

## 4. 总损失与梯度边界

### 4.1 PPO 损失

分别计算教师和学生样本的 PPO surrogate mean：

~~~text
L_ppo = L_ppo_teacher + student_ppo_coef * L_ppo_student
~~~

建议 student_ppo_coef 初始为 1.0。

注意：教师环境数量约为学生的三倍，但两个 group loss 应先分别求均值再相加。若直接对全部样本求一个均值，学生 PPO 信号会被教师样本数量压低。

PPO 优化器只管理：

- teacher_encoder；
- actor；
- critic；
- action std。

学生上下文进入 Actor 前必须 detach：

~~~python
student_context = student_context.detach()
~~~

这样学生轨迹仍然会更新共享 Actor，但 PPO 不会直接破坏 HIM 表征。

### 4.2 HIM 表征损失

学生表征优化器管理：

- him_source_encoder；
- him_target_encoder；
- prototypes。

第一版表征总损失：

~~~text
L_repr =
    velocity_loss_coef * L_velocity
  + him_loss_coef * L_swav
  + align_loss_coef * L_align
~~~

各项定义如下：

~~~text
L_velocity = MSE(predicted_velocity, true_base_velocity)
~~~

~~~text
L_swav = swapped_assignment_loss(z_him, z_next_response)
~~~

~~~text
L_align = 1 - cosine_similarity(z_him, stop_gradient(z_teacher))
~~~

z_him 和 z_teacher 都应做 L2 normalization。

### 4.3 为什么不建议一开始使用强 MSE 对齐

CTS 的 z_teacher 是由策略梯度塑造的特权表征，HIM 的 z_him 是由下一时刻响应和 prototype 塑造的对比表征。两者的坐标系没有天然逐维对应关系。

因此第一版建议：

- 使用 cosine alignment；
- teacher latent 停止梯度；
- align_loss_coef 从较小数值开始；
- 观察 velocity loss 和 SwAV loss 正常后，再逐渐增加对齐强度。

可参考的起始配置，而非最终最优值：

~~~text
velocity_loss_coef = 1.0
him_loss_coef = 1.0
align_loss_coef = 0.1
align_warmup_iterations = 100
teacher_env_ratio = 0.75
num_prototypes = 16
~~~

如果发现加入 alignment 后 terrain level 或 velocity estimation 明显退化，应先把 align_loss_coef 设为 0，验证 CTS + HIM 无对齐版本，而不是同时调整奖励和网络宽度。

### 4.4 后续可升级为三视图原型对齐

第一版稳定后，可以将直接 cosine alignment 升级为三视图对比：

~~~text
history view:       z_him
privileged view:    z_teacher
future response:    z_next
~~~

在共享 prototype 空间中对齐：

- z_him 与 z_next：HIM 原始目标；
- z_him 与 z_teacher：CTS 教师指导；
- 可选 z_teacher 与 z_next：限制教师表征关注机器人响应。

该版本更有研究创新性，但不应作为第一次手动修改的起点。

## 5. 文件修改总览

建议新增以下文件，不要覆盖当前 PPO 文件：

| 文件 | 用途 | 主要参考 |
| --- | --- | --- |
| rsl_rl/rsl_rl/modules/him_estimator.py | HIM source、target、prototype、Sinkhorn | HIMLoco 的 him_estimator.py |
| rsl_rl/rsl_rl/modules/actor_critic_cts_him.py | 教师/学生编码与共享 Actor-Critic | CTS 的 actor_critic_cts.py |
| rsl_rl/rsl_rl/algorithms/cts_him.py | 两组 PPO 和表征更新 | CTS 的 cts.py、HIM 的 him_ppo.py |
| rsl_rl/rsl_rl/storage/rollout_storage_cts_him.py | 保存分组轨迹、历史和下一响应 | CTS/HIM 两种 storage |
| rsl_rl/rsl_rl/runners/on_policy_runner_cts_him.py | 维护 history、收集下一状态 | CTS runner |

建议修改以下文件：

| 文件 | 修改内容 |
| --- | --- |
| rsl_rl/rsl_rl/modules/__init__.py | 导出 HIMEstimator 和 ActorCriticCTSHIM |
| rsl_rl/rsl_rl/algorithms/__init__.py | 导出 CTSHIM |
| rsl_rl/rsl_rl/storage/__init__.py | 导出 RolloutStorageCTSHIM |
| rsl_rl/rsl_rl/runners/__init__.py | 导出 OnPolicyRunnerCTSHIM |
| legged_gym/envs/go2/go2_config.py | 添加 GO2RoughCfgCTSHIM |
| legged_gym/envs/__init__.py | 注册 go2_cts_him |
| legged_gym/utils/task_registry.py | 按配置选择 runner |

训练稳定以后再修改：

| 文件 | 修改内容 |
| --- | --- |
| legged_gym/utils/exporter.py | 导出 student encoder + Actor 的组合模型 |
| legged_gym/scripts/play.py | 建立并维护历史缓冲 |
| deploy/include/control/rl_Inference.h | 输入从 45 改为 225 |
| deploy/src/control/rl_Inference.cpp | 检查并拷贝 225 维输入 |
| deploy/include/FSM/State_Rl.h | 增加当前观测和历史缓冲 |
| deploy/src/FSM/State_Rl.cpp | 每个控制周期更新历史 |
| deploy/configs/config.yaml | model.input_size 改为 225 |

## 6. 分阶段修改路线

### 阶段 0：保存当前 PPO 基线

目标：确保后续所有改动能够与完全相同环境下的 PPO 对比。

操作：

1. 不改变当前 reward、terrain、domain randomization、observation scale 和 action scale。
2. 保存当前 go2 的训练配置和一条短训练曲线。
3. 记录当前 actor/critic 输入维度。
4. 创建独立开发分支。

建议提交：

~~~text
chore: record Go2 PPO fusion baseline
~~~

验收：

- 原来的 go2 task 仍能训练；
- 原来的 checkpoint 仍能加载；
- 当前工作树中已有的部署修改不被融合工作覆盖。

### 阶段 1：只移植 CTS 训练骨架

目标：先让教师/学生并发 PPO 跑起来，不加入 HIM。

#### 6.1 新建 CTS-HIM Actor-Critic 外壳

在 actor_critic_cts_him.py 中先实现：

~~~text
teacher_encoder(privileged_obs) -> 16
temporary_student_encoder(history) -> 16
actor(obs + velocity + latent) -> 12
critic(privileged_obs + velocity + latent) -> 1
~~~

临时学生编码器可以先使用普通 MLP，不做 SwAV。这样可以先验证 CTS 的分组、storage 和 PPO 是否正确。

建议提供清晰接口：

~~~python
encode_teacher(privileged_obs)
encode_student(history)
act_group(obs, privileged_obs, history, is_teacher)
evaluate_group(privileged_obs, history, is_teacher)
act_student(obs, history)
~~~

不要依赖模型内部固定 num_envs 的 history。训练 history 由 runner 管理，部署 history 由 play 或 C++ 管理。模型本身尽量保持无状态。

#### 6.2 教师和学生环境划分

第一版固定使用 3:1：

~~~text
teacher_env_ratio = 0.75
~~~

建议使用交错索引，使教师和学生都能覆盖不同 terrain：

~~~python
student_env_indices = torch.arange(num_envs, device=device)[::4]
teacher_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
teacher_mask[student_env_indices] = False
teacher_env_indices = torch.arange(num_envs, device=device)[teacher_mask]
~~~

不要默认使用连续的前 75% 环境作为教师，因为 Isaac Gym 环境编号可能与 terrain 行列相关，连续切分可能导致两组 terrain 分布不同。

#### 6.3 PPO mini-batch 约束

storage 可以把教师样本和学生样本分别采样，再拼成：

~~~text
[teacher mini-batch, student mini-batch]
~~~

algorithm 必须知道边界位置，分别计算 group mean。

在每个 iteration 输出：

- teacher reward；
- student reward；
- teacher surrogate loss；
- student surrogate loss；
- teacher/student terrain level。

#### 6.4 修改 runner 选择方式

当前 task_registry.py 固定创建 OnPolicyRunner。建议改成显式映射，不直接使用无约束 eval：

~~~python
runner_classes = {
    "OnPolicyRunner": OnPolicyRunner,
    "OnPolicyRunnerCTSHIM": OnPolicyRunnerCTSHIM,
}
runner_class = runner_classes[train_cfg.runner_class_name]
runner = runner_class(env, train_cfg_dict, log_dir, device=args.rl_device)
~~~

这样原始 go2 task 继续使用 OnPolicyRunner，新 task 使用 OnPolicyRunnerCTSHIM。

#### 6.5 阶段 1 验收

- 64 个环境、5 个 iteration 可以完成；
- teacher/student action 均为 [N, 12]；
- teacher/student reward 都在变化；
- PPO loss、value loss、entropy 均为有限值；
- 临时 student latent reconstruction loss 能下降；
- 原始 go2 task 不受影响。

建议提交：

~~~text
feat: add concurrent teacher-student training skeleton
~~~

### 阶段 2：将临时学生编码器替换为 HIM

目标：加入速度估计、future response target 和 prototype contrastive learning。

#### 6.6 HIMEstimator 推荐接口

不要直接复制 HIMEstimator.update 内部自带 optimizer 的写法。建议让 estimator 只负责前向和计算 loss，优化器统一由 CTSHIM 管理：

~~~python
class HIMEstimator(nn.Module):
    def encode_history(self, history):
        # returns predicted_velocity [B, 3], z_him [B, 16]

    def encode_target(self, next_response):
        # returns z_next [B, 16]

    def compute_losses(
        self,
        history,
        velocity_target,
        next_response,
        valid_mask,
    ):
        # returns velocity_loss, swav_loss
~~~

原因：

- 避免一个参数被多个 Adam optimizer 管理；
- algorithm 可以统一控制更新顺序；
- 更容易加入 teacher alignment；
- 更容易保存和恢复 optimizer state。

#### 6.7 Runner 中维护历史

初始化：

~~~python
history = torch.zeros(
    num_envs,
    history_length,
    num_obs,
    device=device,
)
history = torch.cat([history[:, 1:], obs.unsqueeze(1)], dim=1)
~~~

每一步的正确时序：

1. 用当前 obs 和 history 选择 action；
2. env.step(action) 得到 next_obs、next_privileged_obs、done；
3. 将“动作前 history”和“动作后 next target”保存到 transition；
4. 对 done 环境清零 history；
5. 将 next_obs 追加到 history 末尾。

不能在保存 transition 之前就把 next_obs 追加到 source history，否则 source 和 target 会错一帧。

#### 6.8 终止 transition 必须屏蔽

当前环境在 step 内部先 reset，再 compute_observations。因此 done 环境返回的 next_obs 已经属于新 episode，不能作为旧 episode 的 HIM successor target。

第一版最简单可靠的处理：

~~~python
valid_him_target = ~dones.bool()
~~~

velocity、SwAV 和 alignment 是否使用 mask 要分别定义：

- SwAV future target：必须屏蔽 done；
- velocity target：使用当前 privileged_obs 时不需要屏蔽；
- teacher alignment：使用当前 state 时不需要屏蔽。

如果有效 SwAV 样本太少，则跳过该 mini-batch 的 SwAV 更新，不能向 Sinkhorn 传入空张量。

#### 6.9 Storage 增加字段

Transition 和 rollout buffer 至少增加：

~~~text
history
next_response
velocity_target
valid_him_target
~~~

还要保留：

~~~text
observations
privileged_observations
actions
rewards
dones
values
returns
advantages
old_log_prob
old_mu
old_sigma
~~~

#### 6.10 阶段 2 验收

先令 align_loss_coef = 0，仅验证 CTS + HIM：

- velocity MSE 持续下降；
- predicted velocity 三个分量不是常数；
- z_him 的 L2 norm 接近 1；
- prototype 使用率不是长期集中在单个 prototype；
- SwAV loss 有限且不突然变成 NaN；
- 学生 PPO reward 不低于临时编码器版本的随机水平；
- done transition 没有进入 next-response 对比。

建议提交：

~~~text
feat: add HIM response estimator to CTS student
~~~

### 阶段 3：加入教师表征对齐

目标：让 HIM 学生表征获得特权教师指导，同时保留 response representation。

更新顺序建议固定为：

1. 使用 teacher/student 两组样本更新 PPO；
2. 冻结 Actor、Critic 和 teacher_encoder；
3. 重新计算 stop-gradient teacher latent；
4. 更新 HIM source、target 和 prototypes；
5. 清空 rollout storage。

学生表征更新伪代码：

~~~python
with torch.no_grad():
    teacher_z = model.teacher_encoder(privileged_obs_batch)

predicted_velocity, student_z = model.him_estimator.encode_history(
    history_batch
)
target_z = model.him_estimator.encode_target(next_response_batch)

velocity_loss = mse(predicted_velocity, velocity_target_batch)
swav_loss = swapped_assignment_loss(
    student_z[valid_mask],
    target_z[valid_mask],
)
align_loss = (
    1.0
    - cosine_similarity(student_z, teacher_z, dim=-1)
).mean()

repr_loss = (
    velocity_loss_coef * velocity_loss
    + him_loss_coef * swav_loss
    + current_align_coef * align_loss
)
~~~

对齐权重采用 warmup：

~~~text
iteration < warmup: current_align_coef = 0
之后线性增加到 align_loss_coef
~~~

阶段 3 验收：

- teacher latent 不会收到 representation optimizer 梯度；
- student encoder 不会收到 PPO optimizer 梯度；
- 两个 optimizer 的参数集合没有交集；
- alignment 加入后 velocity loss 和 prototype 使用率没有明显恶化；
- student reward 或 tracking 至少不比 align=0 明显下降。

建议提交：

~~~text
feat: align CTS teacher and HIM student representations
~~~

### 阶段 4：配置和任务注册

#### 6.11 新训练配置

在 go2_config.py 中新增 GO2RoughCfgCTSHIM，保留 GO2RoughCfg 作为环境配置。

建议结构：

~~~python
class GO2RoughCfgCTSHIM(LeggedRobotCfgPPO):
    runner_class_name = "OnPolicyRunnerCTSHIM"
    history_length = 5

    class policy(LeggedRobotCfgPPO.policy):
        latent_dim = 16
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        teacher_encoder_hidden_dims = [512, 256]
        him_encoder_hidden_dims = [512, 256, 128]
        him_target_hidden_dims = [128, 64]
        num_prototypes = 16

    class algorithm(LeggedRobotCfgPPO.algorithm):
        teacher_env_ratio = 0.75
        student_encoder_learning_rate = 1.0e-3
        velocity_loss_coef = 1.0
        him_loss_coef = 1.0
        align_loss_coef = 0.1
        align_warmup_iterations = 100

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticCTSHIM"
        algorithm_class_name = "CTSHIM"
        experiment_name = "go2_cts_him"
~~~

#### 6.12 注册独立任务

在 legged_gym/envs/__init__.py 注册：

~~~python
task_registry.register(
    "go2_cts_him",
    LeggedRobot,
    GO2RoughCfg(),
    GO2RoughCfgCTSHIM(),
)
~~~

不要把现有 go2 task 指向新配置。这样 PPO baseline 和融合算法能够长期共存。

#### 6.13 日志必须增加

至少记录：

- Loss/value_function；
- Loss/surrogate_teacher；
- Loss/surrogate_student；
- Loss/velocity_estimation；
- Loss/swav；
- Loss/teacher_alignment；
- Policy/entropy；
- Representation/prototype_perplexity；
- Representation/predicted_velocity_error_x/y/z；
- Train/teacher_reward；
- Train/student_reward；
- Train/teacher_terrain_level；
- Train/student_terrain_level。

prototype perplexity 可以帮助发现表征塌缩。仅观察 SwAV loss 不足以判断 prototype 是否被均匀使用。

## 7. 推荐实验顺序

不要直接比较最终 full model。建议按以下顺序保存配置和 checkpoint：

| 实验 | CTS | 速度估计 | SwAV | 教师对齐 |
| --- | ---: | ---: | ---: | ---: |
| PPO baseline | 否 | 否 | 否 | 否 |
| CTS | 是 | 否 | 否 | MSE 临时学生 |
| CTS + velocity | 是 | 是 | 否 | 否 |
| CTS + HIM | 是 | 是 | 是 | 否 |
| CTS + HIM + align | 是 | 是 | 是 | cosine |
| CTS + HIM + prototype align | 是 | 是 | 是 | prototype |

所有实验保持：

- 同一 reward；
- 同一 observation scaling；
- 同一 terrain curriculum；
- 同一 domain randomization；
- 同一 action scale；
- 同一训练步数；
- 至少 3 个随机种子。

优先比较：

- 线速度和角速度 tracking error；
- 最大 terrain level；
- 随机推力后的存活率；
- 学生与教师回报差距；
- velocity estimation error；
- prototype 使用率。

## 8. 推理、导出与部署

这一阶段必须等训练端稳定以后再开始。

### 8.1 部署模型只保留学生路径

训练时使用的以下模块不进入部署：

- teacher_encoder；
- critic；
- him_target_encoder；
- prototypes；
- PPO action distribution。

部署模型只需要：

~~~text
flattened_history -> him_source_encoder -> velocity, z_him
last_obs + velocity + z_him -> actor -> actions
~~~

建议导出一个无状态组合模块：

~~~python
class CTSHIMDeploymentPolicy(nn.Module):
    def __init__(self, source_encoder, actor, history_length, obs_dim):
        ...

    def forward(self, flattened_history):
        current_obs = flattened_history[:, -self.obs_dim:]
        velocity, latent = self.source_encoder(flattened_history)
        actor_input = torch.cat(
            [current_obs, velocity, latent],
            dim=-1,
        )
        return self.actor(actor_input)
~~~

ONNX/MNN 输入输出固定为：

~~~text
input:  [batch, 225]
output: [batch, 12]
~~~

不能继续使用当前 exporter.py 中“只导出 actor”的路径。只导出 Actor 会得到 64 维输入模型，但部署端无法自行生成 19 维 HIM context。

### 8.2 Python play

play.py 中维护：

~~~text
history shape = [num_envs, 5, 45]
~~~

每步：

1. 更新 obs 中的手柄 command；
2. history 左移一帧；
3. 将当前 obs 放入最后一帧；
4. 将 history flatten 后送入部署组合策略；
5. done 环境清零历史。

### 8.3 C++ MNN

当前 C++ 只接受 45 维输入，需要同步修改：

- 当前一步观测仍保持 45 维；
- 新增 225 维 history 数组；
- State_Rl::enter 中将 history 清零；
- 每次 getObservation 后左移 45 个元素并追加当前观测；
- mnnInference 将 history 传给 MNN；
- MNN input element size 检查改为 225；
- config.yaml 的 model.input_size 改为 225。

训练和 C++ 必须使用完全相同的历史顺序：

~~~text
oldest -> newest
~~~

第一次进入 RL 状态时使用零历史，逐帧填充，与训练 episode reset 后的处理保持一致。

建议提交：

~~~text
feat: export and deploy history-based CTS-HIM policy
~~~

## 9. 验证命令

### 9.1 语法检查

~~~bash
python -m py_compile \
  rsl_rl/rsl_rl/modules/him_estimator.py \
  rsl_rl/rsl_rl/modules/actor_critic_cts_him.py \
  rsl_rl/rsl_rl/algorithms/cts_him.py \
  rsl_rl/rsl_rl/storage/rollout_storage_cts_him.py \
  rsl_rl/rsl_rl/runners/on_policy_runner_cts_him.py
~~~

### 9.2 小规模训练

~~~bash
python legged_gym/scripts/train.py \
  --task=go2_cts_him \
  --headless \
  --num_envs=64 \
  --max_iterations=5
~~~

小规模训练只验证接口和数值稳定性，不用于判断算法效果。Sinkhorn 对 batch diversity 敏感，因此正式训练仍需要较多并行环境。

### 9.3 完整训练

~~~bash
python legged_gym/scripts/train.py \
  --task=go2_cts_him \
  --headless
~~~

### 9.4 导出后形状检查

至少检查：

~~~text
ONNX input  = [batch, 225]
ONNX output = [batch, 12]
~~~

并用同一组 225 维输入比较 PyTorch 与 ONNX 输出的最大绝对误差。

### 9.5 C++ 编译

~~~bash
cmake -S deploy -B deploy/build
cmake --build deploy/build -j
~~~

C++ 修改必须保持 -Wall -Wextra -Wpedantic 下无新增警告。

## 10. 常见错误排查

### 10.1 学生 reward 一直不增长

依次检查：

1. student PPO loss 是否真的加入总 loss；
2. student_context 是否只 detach 表征，而不是把整个 Actor 输出 detach；
3. student mini-batch 是否为空；
4. teacher/student 样本边界是否正确；
5. student history 是否包含当前 obs。

### 10.2 Velocity loss 不下降

检查：

- velocity_target 是否为 privileged_obs 的 45:48；
- lin_vel scale 是否与训练观测一致；
- history 顺序是否 oldest-to-newest；
- done 后 history 是否清零；
- source encoder 是否被 representation optimizer 管理。

### 10.3 SwAV 出现 NaN

检查：

- prototype 是否每次计算前 L2 normalize；
- Sinkhorn 输入是否为空；
- batch 中有效 transition 数量是否过少；
- exp 前的 temperature/epsilon 是否过小；
- z_him 和 z_next 是否已经 normalize；
- 是否错误地将 done 后 reset observation 当作 next target。

### 10.4 加入教师对齐后性能下降

按顺序处理：

1. align_loss_coef 设为 0，确认 CTS + HIM 本身正常；
2. 确认 teacher_z 使用 stop-gradient；
3. 将 MSE 改为 cosine；
4. 延长 warmup；
5. 降低 align_loss_coef；
6. 最后再考虑 prototype-level alignment。

不要通过修改 reward 来掩盖 representation loss 冲突。

### 10.5 仿真正常但 MNN 失败

检查：

- 导出的是否为 source encoder + Actor，而不是单独 Actor；
- ONNX/MNN 输入是否为 225；
- C++ history 顺序是否一致；
- observation ordering 和 scale 是否仍为当前仓库的 45 维定义；
- command 是否写入 6:9；
- previous action 是否写入 33:45；
- episode/状态切换时历史是否清零。

## 11. 推荐提交顺序

每个提交只完成一种行为：

~~~text
chore: record Go2 PPO fusion baseline
feat: add concurrent teacher-student training skeleton
feat: add HIM response estimator to CTS student
feat: align CTS teacher and HIM student representations
feat: add CTS-HIM task configuration and logging
feat: export and deploy history-based CTS-HIM policy
~~~

不要在同一个提交中同时修改 reward、terrain、算法网络和部署输入，否则训练退化时无法定位原因。

## 12. 完成标准

训练端完成：

- go2 和 go2_cts_him 可以独立注册和训练；
- teacher/student PPO 都产生有效梯度；
- student encoder 只由 representation optimizer 更新；
- velocity、SwAV、alignment loss 都被记录；
- done transition 不进入 future-response loss；
- checkpoint 可以恢复两个 optimizer。

算法验证完成：

- 至少完成 PPO、CTS、CTS + HIM、完整融合四组对比；
- 至少 3 个随机种子；
- tracking、terrain、push robustness 和 representation 指标都有记录。

部署端完成：

- 导出模型为 225 输入、12 输出；
- PyTorch、ONNX、MNN 的同输入输出一致；
- C++ 历史顺序和训练完全一致；
- MuJoCo 中不存在输入维度、NaN 或动作顺序错误。

## 13. 参考代码定位

HIM 核心：

- /home/asuka/Legged/HIMLoco/rsl_rl/rsl_rl/modules/him_estimator.py
- /home/asuka/Legged/HIMLoco/rsl_rl/rsl_rl/modules/him_actor_critic.py
- /home/asuka/Legged/HIMLoco/rsl_rl/rsl_rl/algorithms/him_ppo.py
- /home/asuka/Legged/HIMLoco/rsl_rl/rsl_rl/storage/him_rollout_storage.py
- /home/asuka/Legged/HIMLoco/rsl_rl/rsl_rl/runners/him_on_policy_runner.py

CTS 核心：

- /home/asuka/Legged/go2_rl_gym/rsl_rl/rsl_rl/modules/actor_critic_cts.py
- /home/asuka/Legged/go2_rl_gym/rsl_rl/rsl_rl/algorithms/cts.py
- /home/asuka/Legged/go2_rl_gym/rsl_rl/rsl_rl/storage/rollout_storage_cts.py
- /home/asuka/Legged/go2_rl_gym/rsl_rl/rsl_rl/runners/on_policy_runner_cts.py

参考代码不能整文件覆盖当前仓库。尤其需要重新核对：

- observation ordering；
- privileged observation ordering；
- history ordering；
- terminal observation；
- runner 创建方式；
- exporter 输入；
- C++ 部署维度。
