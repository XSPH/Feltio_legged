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

本文只规划修改，不要求一次完成全部代码。教师对齐采用渐进路线：先完成无对齐的
CTS + HIM 聚类基线，再实现简单的 cosine latent 对齐，最后实现 prototype-level
分配对齐。cosine 和 prototype 是两种独立对照方案，不在同一次正式实验中叠加。
建议严格按照阶段顺序推进，每个阶段通过验收后再进入下一阶段。

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
- 可选的 teacher_projection。

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
- prototypes；
- teacher_projection（仅 prototype alignment 模式）。

第一步只训练 HIM 自身的速度估计和聚类表征，不加入教师对齐：

~~~text
L_repr_base =
    velocity_loss_coef * L_velocity
  + him_loss_coef * L_swav
~~~

各项定义如下：

~~~text
L_velocity = MSE(predicted_velocity, true_base_velocity)
~~~

~~~text
L_swav = swapped_assignment_loss(z_him, z_next_response)
~~~

其中 L_swav 必须保留 HIM 的完整 prototype、Sinkhorn balanced assignment 和
swapped-assignment 计算。该损失使 history view 和 future-response view 获得一致且相对
均衡的 prototype 分配，是本融合方案保留 HIM 聚类能力的核心。

教师对齐只作为 L_repr_base 之上的可选实验项。第一版训练必须令
alignment_type = "none"，确认 HIM 聚类稳定后，再分别测试 cosine 和 prototype 两种
对齐方式。两种对齐不能在同一组对比实验中同时开启。

### 4.3 第一版简单方案：cosine latent 对齐

CTS 的 z_teacher 是由策略梯度塑造的特权表征，HIM 的 z_him 是由下一时刻响应和 prototype 塑造的对比表征。两者的坐标系没有天然逐维对应关系。

直接 MSE 要求逐维数值一致，不适合这里的异构表征。为了先从简单方案开始，可在
CTS + HIM 基线稳定后增加 cosine alignment：

~~~text
L_align_cosine =
    1 - cosine_similarity(z_him, stop_gradient(z_teacher))

L_repr_cosine = L_repr_base + align_loss_coef * L_align_cosine
~~~

z_him 和 z_teacher 都应先做 L2 normalization。该方案实现简单，适合用于快速验证
教师指导是否有效，但它仍然在对齐语义不同的 latent，可能压坏 HIM 已形成的聚类结构，
因此只能作为第一版方案和后续消融基线，不能预先假定它一定优于无对齐版本。

实现要求：

- teacher latent 停止梯度；
- align_loss_coef 从较小数值开始，并在 warmup 前保持为 0；
- 观察 velocity loss 和 SwAV loss 正常后，再逐渐增加对齐强度。

可参考的起始配置，而非最终最优值：

~~~text
velocity_loss_coef = 1.0
him_loss_coef = 1.0
alignment_type = "none"
align_loss_coef = 0.1  # 仅 alignment_type="cosine" 时生效
align_warmup_iterations = 100
teacher_env_ratio = 0.75
num_prototypes = 16
~~~

如果加入 cosine alignment 后 terrain level、velocity estimation 或 prototype perplexity
明显退化，应切回同配置的 alignment_type = "none" 基线定位问题，而不是同时调整奖励、
网络宽度或聚类超参数。

### 4.4 第二版升级方案：prototype 分配对齐

如果实验目标是最大限度保留 HIM 的聚类分配优势，应在 cosine 方案完成后，单独实现
prototype-level alignment。三种视图为：

~~~text
history view:       z_him
privileged view:    z_teacher_projected
future response:    z_next
~~~

teacher_encoder 的输出不能直接与 HIM prototype 计算相似度，必须先经过独立的
teacher_projection，并保持 teacher_encoder stop-gradient。teacher_projection 由
representation optimizer 管理，用 next-response 的分配训练它预测同一聚类语义：

~~~text
q_next    = sinkhorn(score(z_next, prototypes))
p_teacher = softmax(score(
    teacher_projection(stopgrad(z_teacher)), stopgrad(prototypes)
))
L_teacher_prototype = CE(stopgrad(q_next), p_teacher)
~~~

教师相关 loss 中使用 stop-gradient prototype weights，使 prototypes 仍然只由 HIM
原始 L_swav 更新。这样教师负责学习和指导已有聚类分配，而不会反过来任意重塑 HIM
的聚类中心。

teacher_projection 稳定后，再让教师的平衡分配指导学生：

~~~text
q_teacher = sinkhorn(score(z_teacher_projected, prototypes))
p_him     = softmax(score(z_him, prototypes))
L_prototype_align = CE(stopgrad(q_teacher), p_him)

L_repr_prototype =
    L_repr_base
  + teacher_prototype_loss_coef * L_teacher_prototype
  + prototype_align_coef * L_prototype_align
~~~

这样对齐的是“样本属于哪个隐式响应簇”，而不是强迫两个 latent 的坐标逐维一致。在
共享 prototype 空间中的关系为：

- z_him 与 z_next：HIM 原始目标；
- z_teacher_projected 与 z_next：训练教师投影预测响应簇；
- z_him 与 z_teacher_projected：CTS 教师提供 prototype 分配指导。

该版本更符合保留 HIM 聚类能力的目标，但实现和调参成本更高，不作为第一次手动修改的
起点。它应与 cosine 方案分别从相同 CTS + HIM 基线出发进行比较，不能把 prototype
alignment 叠加在 cosine alignment 上后再声称两者是公平对比。

## 5. 文件修改总览

当前工作在独立 cts 分支中进行，保留现有算法、类和任务名称，直接扩展以下训练文件：

| 文件 | 修改内容 | 主要参考 |
| --- | --- | --- |
| rsl_rl/rsl_rl/modules/actor_critic.py | 教师/学生编码与共享 Actor-Critic | CTS 的 actor_critic_cts.py |
| rsl_rl/rsl_rl/algorithms/ppo.py | 两组 PPO 和表征更新 | CTS 的 cts.py、HIM 的 him_ppo.py |
| rsl_rl/rsl_rl/storage/rollout_storage.py | 保存分组轨迹、历史和下一响应 | CTS/HIM 两种 storage |
| rsl_rl/rsl_rl/runners/on_policy_runner.py | 维护 history、收集下一状态 | CTS runner |
| legged_gym/envs/base/legged_robot_config.py | 配置历史长度、编码器和分组 PPO 参数 | 当前 PPO 配置 |

阶段二只新增 rsl_rl/rsl_rl/modules/him_estimator.py，用于 HIM source、target、prototype
和 Sinkhorn；ActorCritic、PPO、RolloutStorage、OnPolicyRunner 继续沿用现有名称。

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

#### 6.1 改造现有 Actor-Critic 外壳

在 actor_critic.py 中先实现：

~~~text
teacher_encoder(privileged_obs) -> 16
teacher_context = true_base_velocity (3) + teacher_latent (16)
temporary_student_encoder(history) -> predicted_velocity (3) + student_latent (16)
actor(obs + velocity + latent) -> 12
critic(privileged_obs + velocity + latent) -> 1
~~~

临时学生编码器可以先使用普通 MLP 输出 19 维 context，不做 SwAV。其前 3 维作为临时
速度预测，后 16 维做 L2 normalization 后作为 student latent。这样阶段一就固定 Actor
的 64 维输入，阶段二替换为 HIMEstimator 时无需改变 Actor 接口，同时可以先验证 CTS
的分组、storage 和 PPO 是否正确。

建议提供清晰接口：

~~~python
encode_teacher(privileged_obs)
encode_student(history)
act_group(obs, privileged_obs, history, is_teacher)
evaluate_group(privileged_obs, history, is_teacher)
act_student(obs, history)
~~~

不要依赖模型内部固定 num_envs 的 history。训练 history 由 runner 管理，Python play
和 C++ 部署端分别维护自己的 history，模型本身保持无状态。所有路径统一使用“最旧帧在前、
当前帧在最后”的排列，不直接照搬 HIMLoco 中“当前帧在最前”的切片方式。

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

阶段一使用两个互不重叠的优化器：

- PPO optimizer：teacher_encoder、Actor、Critic 和 action std；
- temporary student optimizer：temporary_student_encoder。

临时学生只在 student mini-batch 上蒸馏教师的完整 19 维 context：

~~~text
teacher_context = concat(true_base_velocity, teacher_latent)
student_context = concat(predicted_velocity, student_latent)
L_temporary_student = MSE(
    student_context,
    stop_gradient(teacher_context),
)
~~~

student_context 进入共享 Actor 和 Critic 前保持 detach。这样学生轨迹仍通过
L_ppo_student 更新共享策略，但 PPO 不会绕过蒸馏目标直接更新临时学生编码器。

在每个 iteration 输出：

- teacher reward；
- student reward；
- teacher surrogate loss；
- student surrogate loss；
- temporary student context loss；
- teacher/student terrain level。

#### 6.4 保持现有 runner 和 task 名称

当前实现位于独立 cts 分支，继续使用 OnPolicyRunner、ActorCritic、PPO、RolloutStorage
和 go2 名称，不修改 task_registry，也不新增并行注册。main 分支保留原始 PPO 基线。

#### 6.5 阶段 1 验收

- 64 个环境、5 个 iteration 可以完成；
- teacher/student action 均为 [N, 12]；
- teacher/student reward 都在变化；
- PPO loss、value loss、entropy 均为有限值；
- 临时 student latent reconstruction loss 能下降；
- cts 分支中的 go2 task 可以完成阶段一训练。

建议提交：

~~~text
feat: add concurrent teacher-student training skeleton
~~~

### 阶段 2：将临时学生编码器替换为 HIM

目标：加入速度估计、future response target 和 prototype contrastive learning。

#### 6.6 HIMEstimator 推荐接口

不要直接复制 HIMEstimator.update 内部自带 optimizer 的写法。建议让 estimator 只负责前向和计算 loss，优化器统一由 PPO 管理：

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

1. 用当前 obs 和 history 选择 action，并在 transition 中保存动作前 history；
2. env.step(action) 得到 next_obs、next_privileged_obs、done；
3. algorithm 保存已捕获的动作前 history 和动作后的 next target；
4. 对 done 环境清零 history；
5. 将 next_obs 追加到 history 末尾。

不能在保存 transition 之前就把 next_obs 追加到 source history，否则 source 和 target
会错一帧。HIMEstimator 接收该排列时，当前帧应从 history 的最后 45 维读取。

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

### 阶段 3A：加入简单的 cosine 教师对齐

目标：用最小改动验证特权教师指导是否有益。该阶段以阶段 2 的 CTS + HIM 为共同基线，
只加入 cosine alignment，不加入 teacher_projection 或 prototype alignment。

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

阶段 3A 验收：

- teacher latent 不会收到 representation optimizer 梯度；
- student encoder 不会收到 PPO optimizer 梯度；
- 两个 optimizer 的参数集合没有交集；
- cosine alignment 加入后 velocity loss、prototype perplexity 和分配熵没有明显恶化；
- student reward 或 tracking 至少不比 align=0 明显下降。

建议提交：

~~~text
feat: align CTS teacher and HIM student representations
~~~

### 阶段 3B：加入 prototype-level 教师对齐

目标：在不要求 teacher/student latent 逐维一致的前提下，让教师通过 prototype 分配
指导 HIM 学生，并与阶段 3A 做独立对比。

不要在阶段 3A 的 cosine loss 上继续叠加本方案。调试时可以从同一个阶段 2 checkpoint
分别创建两个分支；正式实验应使用相同配置和随机种子分别从头训练。

#### 6.11 增加教师投影头

在 ActorCritic 中增加 teacher_projection，将 stop-gradient teacher latent
映射到 HIM prototype 空间。teacher_projection 由 representation optimizer 管理，
teacher_encoder 仍只由 PPO optimizer 管理。

训练顺序：

1. 正常计算 HIM 的 history/future-response swapped-assignment loss；
2. 用 q_next 监督 p_teacher，先让 teacher_projection 学会预测 future-response cluster；
3. warmup 结束且 teacher prototype prediction 不再接近随机后，用 q_teacher 指导 p_him；
4. teacher_encoder 在两个 prototype loss 中都保持 stop-gradient。

prototype 对齐伪代码：

~~~python
with torch.no_grad():
    teacher_z = model.teacher_encoder(privileged_obs_batch)

teacher_projected_z = normalize(model.teacher_projection(teacher_z))
student_z = normalize(student_z)
target_z = normalize(target_z)

score_student = student_z @ prototypes.T  # 用于原始 L_swav
score_target = target_z @ prototypes.T

prototype_targets = prototypes.detach()
score_student_align = student_z @ prototype_targets.T
score_teacher = teacher_projected_z @ prototype_targets.T

with torch.no_grad():
    q_target = sinkhorn(score_target)
    q_teacher = sinkhorn(score_teacher)

p_teacher = softmax(score_teacher / temperature, dim=-1)
p_student = softmax(score_student_align / temperature, dim=-1)

teacher_prototype_loss = cross_entropy(q_target, p_teacher)
prototype_align_loss = cross_entropy(q_teacher, p_student)
~~~

这里的 cross_entropy 表示 soft-target cross entropy。prototype_align_loss 只能在
teacher_projection warmup 后启用。

阶段 3B 验收：

- teacher_projection 收到 representation gradient，teacher_encoder 不收到该梯度；
- teacher/target prototype agreement 或交叉熵明显优于均匀随机基线；
- prototype 使用率、perplexity 和分配熵没有塌缩；
- 与 cosine 方案相比，使用相同训练预算、环境配置和随机种子；
- 不以单次训练结果判断优劣。

建议提交：

~~~text
feat: add prototype-level teacher alignment
~~~

### 阶段 4：配置和日志

#### 6.12 扩展现有训练配置

继续扩展 legged_robot_config.py 中的 LeggedRobotCfgPPO，不新增配置类或修改算法类名。

建议结构：

~~~python
class LeggedRobotCfgPPO(BaseConfig):
    runner_class_name = "OnPolicyRunner"
    history_length = 5

    class policy:
        latent_dim = 16
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        teacher_encoder_hidden_dims = [512, 256]
        him_encoder_hidden_dims = [512, 256, 128]
        him_target_hidden_dims = [128, 64]
        num_prototypes = 16

    class algorithm:
        teacher_env_ratio = 0.75
        student_encoder_learning_rate = 1.0e-3
        velocity_loss_coef = 1.0
        him_loss_coef = 1.0
        alignment_type = "none"  # "none", "cosine", "prototype"
        align_loss_coef = 0.1
        align_warmup_iterations = 100
        teacher_prototype_loss_coef = 1.0
        prototype_align_coef = 0.1
        prototype_align_warmup_iterations = 100

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
~~~

alignment_type 必须保证两种教师对齐互斥：

- none：只训练 CTS + HIM，所有教师对齐系数视为 0；
- cosine：只启用 align_loss_coef；
- prototype：只启用 teacher_prototype_loss_coef 和 prototype_align_coef。

#### 6.13 保持现有任务注册

继续使用 legged_gym/envs/__init__.py 中已有的 go2 注册：

~~~python
task_registry.register(
    "go2",
    LeggedRobot,
    GO2RoughCfg(),
    GO2RoughCfgPPO(),
)
~~~

不新增 task_registry 映射。原始 PPO 基线由 main 分支保留，cts 分支中的 go2 使用融合算法。

#### 6.14 日志必须增加

至少记录：

- Loss/value_function；
- Loss/surrogate_teacher；
- Loss/surrogate_student；
- Loss/velocity_estimation；
- Loss/swav；
- Loss/teacher_alignment_cosine；
- Loss/teacher_prototype_prediction；
- Loss/teacher_alignment_prototype；
- Policy/entropy；
- Representation/prototype_perplexity；
- Representation/prototype_assignment_entropy；
- Representation/prototype_usage_min/max；
- Representation/teacher_target_prototype_agreement；
- Representation/predicted_velocity_error_x/y/z；
- Train/teacher_reward；
- Train/student_reward；
- Train/teacher_terrain_level；
- Train/student_terrain_level。

prototype perplexity、分配熵和各 prototype 使用率可以帮助发现表征塌缩。仅观察
SwAV loss 不足以判断 prototype 是否被均匀使用。未启用的 alignment loss 应记录为 0，
同时记录 alignment_type，避免比较实验时混淆实际生效的方案。

## 7. 推荐实验顺序

不要直接比较最终 full model。先建立共同的 CTS + HIM 基线，再让两种教师对齐方案从
该基线独立分叉。建议按以下顺序保存配置和 checkpoint：

| 实验 | CTS | 速度估计 | SwAV | alignment_type | 目的 |
| --- | ---: | ---: | ---: | --- | --- |
| PPO baseline | 否 | 否 | 否 | none | 原始策略基线 |
| CTS | 是 | 否 | 否 | CTS 临时学生 MSE | 验证双分支 PPO |
| CTS + velocity | 是 | 是 | 否 | none | 隔离速度估计收益 |
| CTS + HIM | 是 | 是 | 是 | none | 聚类融合共同基线 |
| CTS + HIM + cosine | 是 | 是 | 是 | cosine | 第一版简单教师对齐 |
| CTS + HIM + prototype | 是 | 是 | 是 | prototype | 聚类分配教师对齐 |

所有实验保持：

- 同一 reward；
- 同一 observation scaling；
- 同一 terrain curriculum；
- 同一 domain randomization；
- 同一 action scale；
- 同一网络宽度和 prototype 数量；
- 同一 batch size、并行环境数和训练步数；
- 至少 3 个随机种子。

cosine 和 prototype 两组必须使用相同的随机种子集合。调试阶段可以从同一个 CTS + HIM
checkpoint 分叉以节省时间；正式结论必须分别从头训练，报告至少 3 个种子的均值和标准差，
不能只比较各自最好的 checkpoint。

优先比较：

- 线速度和角速度 tracking error；
- 最大 terrain level；
- 随机推力后的存活率；
- 学生与教师回报差距；
- velocity estimation error；
- prototype perplexity、分配熵和使用率；
- teacher/target prototype agreement；
- 是否出现 NaN、单簇占用或表征塌缩。

## 8. 推理、导出与部署

这一阶段必须等训练端稳定以后再开始。

### 8.1 部署模型只保留学生路径

训练时使用的以下模块不进入部署：

- teacher_encoder；
- critic；
- him_target_encoder；
- prototypes；
- PPO action distribution。

prototype 和 Sinkhorn 只负责训练阶段塑造 z_him 的聚类结构。部署时 Actor 继续接收经过
该聚类目标训练的连续 z_him，因此仍然保留聚类学习带来的表征收益。不要在单机器人或
batch size 为 1 的推理路径中运行 Sinkhorn；其平衡分配依赖 batch 统计，训练和部署行为
会不一致。

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
4. 将当前 obs 和 history 送入学生策略；
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
  rsl_rl/rsl_rl/modules/actor_critic.py \
  rsl_rl/rsl_rl/algorithms/ppo.py \
  rsl_rl/rsl_rl/storage/rollout_storage.py \
  rsl_rl/rsl_rl/runners/on_policy_runner.py
~~~

### 9.2 小规模训练

~~~bash
python legged_gym/scripts/train.py \
  --task=go2 \
  --headless \
  --num_envs=64 \
  --max_iterations=5
~~~

小规模训练只验证接口和数值稳定性，不用于判断算法效果。Sinkhorn 对 batch diversity 敏感，因此正式训练仍需要较多并行环境。

### 9.3 完整训练

~~~bash
python legged_gym/scripts/train.py \
  --task=go2 \
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

先将 alignment_type 设为 none，确认同配置 CTS + HIM 基线正常，然后按当前方案分别排查。

cosine 方案：

1. 确认 teacher_z 使用 stop-gradient；
2. 确认 z_him 和 z_teacher 均已 L2 normalize；
3. 延长 align_warmup_iterations；
4. 降低 align_loss_coef；
5. 检查 prototype perplexity 是否在启用 cosine 后下降。

prototype 方案：

1. 确认 teacher_encoder stop-gradient，而 teacher_projection 能收到梯度；
2. 确认 teacher_projection 先由 q_next 监督完成 warmup；
3. 检查 teacher/target prototype agreement 是否优于随机水平；
4. 延长 prototype_align_warmup_iterations；
5. 降低 prototype_align_coef，并检查各 prototype 使用率。

不要同时开启 cosine 和 prototype 对齐，也不要通过修改 reward 来掩盖 representation
loss 冲突。

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
feat: add prototype-level teacher alignment
feat: add CTS-HIM configuration and logging
feat: export and deploy history-based CTS-HIM policy
~~~

不要在同一个提交中同时修改 reward、terrain、算法网络和部署输入，否则训练退化时无法定位原因。

## 12. 完成标准

训练端完成：

- cts 分支中的 go2 可以完成融合算法训练；
- teacher/student PPO 都产生有效梯度；
- student encoder 只由 representation optimizer 更新；
- velocity、SwAV、cosine alignment 和 prototype alignment 指标按启用模式正确记录；
- alignment_type 保证 none、cosine、prototype 三种模式互斥；
- done transition 不进入 future-response loss；
- checkpoint 可以恢复 PPO 和 representation 两个 optimizer 的状态；prototype 模式下
  teacher_projection 必须包含在 representation optimizer 及其 checkpoint 中。

算法验证完成：

- 至少完成 PPO、CTS、CTS + HIM、cosine alignment 和 prototype alignment 五组对比；
- 至少 3 个随机种子；
- cosine 与 prototype 使用相同随机种子、训练预算和环境配置；
- tracking、terrain、push robustness、聚类分配和表征指标都有记录。

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
