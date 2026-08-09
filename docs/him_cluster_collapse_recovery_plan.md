# HIM 表征聚类坍缩修复与逐步验证方案

## 1. 目标

本方案用于定位并修复当前 HIM student latent 和 prototypes 的表示坍缩问题。执行时必须一次只改变一个训练因素，并且每个实验都从随机初始化开始训练，禁止从已经坍缩的 checkpoint 继续训练。

### 分支用途

`him-collapse-step0` 是一次性诊断分支，只用于增加观测指标、执行消融实验和收集聚类坍缩证据。该分支禁止合并到 `hcts`。

找到并验证根因后，执行以下流程：

1. 记录能够复现根因和解除坍缩的最小实验差异；
2. 切回 `hcts`，不合并 `him-collapse-step0`；
3. 根据已确认的原因，在 `hcts` 上重新实现最小生产修复；
4. 只保留生产训练确实需要的代码，不默认带入临时诊断、短实验配置和消融开关；
5. 在 `hcts` 上重新运行 smoke test、100 轮验证和必要的多 seed 验证。

最终目标不是让 t-SNE 图“看起来分得开”，而是先满足以下高维空间条件：

1. prototypes 不再退化为一条直线或正反两个方向；
2. source latent 和 target latent 保持多个有效维度；
3. prototype 的硬分配不会长期集中到极少数 prototype；
4. velocity estimation、PPO reward 和地形课程学习没有明显退化；
5. 高维指标稳定后，才使用 t-SNE 和地形标签检查可视化结果。

## 2. 当前已经确认的现象

现有 checkpoint 已经证明这不是单纯的 t-SNE 参数或训练轮数问题：

- `Aug09_11-50-09` 中，prototype effective rank 从初始化附近的 `9.391` 降到第 100 轮附近的 `1.116`；
- `Aug08_11-36-26` 中，从第 1000 轮到第 20000 轮，prototype effective rank 一直约为 `1`；
- 最终 16 个 prototypes 退化为同一条轴上的两个方向，其中约 13 个为一个方向、3 个为相反方向；
- 已保存的 student latent 随机样本对余弦相似度均值约为 `0.9871`，中心化 effective rank 约为 `1.48`；
- 当前温度 `him_temperature=3.0` 下，13/3 的正反 prototype 结构产生的 soft usage 与训练日志高度吻合；
- 更换梯度裁剪、环境数量、rollout 长度、更新是否流式、训练轮数和地形等级后仍出现坍缩。

因此，后续实验不再优先调整以下因素：

- t-SNE perplexity；
- PPO/HIM 梯度裁剪是否分别设置；
- 2048 或 4096 个环境；
- 24 或 100 步 rollout；
- 是否训练到更高地形等级；
- 是否把所有 mini-batch 先保存为列表。

## 3. 当前因果判断与下一假设

E1 已经验证“teacher 样本混入 HIM”不是主要根因。把 representation update
从 teacher/student 淈合样本改为 student-only 后，坍缩仅被轻微延后：第 100
轮 prototype effective rank 仍降到 `1.055`，student latent 的随机样本对余弦
相似度升到 `0.9991`，最大 hard prototype usage 达到 `0.9717`。teacher 和
student 两组 latent 最终也同时坍缩。

但是，E1 的 100 轮实验发生在机器人尚未形成稳定步态、terrain level 仍为 0
的阶段，因此它只能证明“早期已经坍缩”，不能单独回答 student-only 表示在策略
成熟后是否可能恢复。开始 E2 前，先执行 E1-long：保持 student-only，不使用
prototype freeze，改用生产训练的 24 步 rollout，从随机初始化训练 10000 轮。

如果 E1-long 在稳定步态和多地形课程出现后仍保持坍缩，下一项待验证假设才是
训练初期 encoder 与随机 prototypes 同时快速追逐，导致系统进入共线局部解。
支持该候选假设的现有现象是：

- E0/E1 的 prototype effective rank 都在前 20 至 40 轮快速下降；
- E1 的 sample mode 改动没有改变最终的 4/12 正反方向 prototype 结构；
- 第 100 轮 target encoder 和 prototype 梯度已经很小，说明进入坍缩吸引子后，
  继续增加轮数很难自行脱离。

E2 暂停，等待 E1-long 数据。若后续执行，E2 只冻结最初 300 次 HIM optimizer
step 的 prototype 参数，encoder 和 velocity head 继续正常学习。该实验仍不改
temperature、Sinkhorn epsilon、loss reduction、优化器或 PPO 更新顺序。

## 4. 固定实验条件

所有短实验统一使用同一组条件。推荐使用最近已经能在当前显卡运行的诊断配置：

| 配置项 | 固定值 |
| --- | --- |
| `num_envs` | 2048 |
| `num_steps_per_env` | 100 |
| `max_iterations` | 100 |
| `save_interval` | 10 |
| `seed` | 固定为同一个值 |
| `teacher_env_ratio` | 0.75 |
| `num_prototypes` | 16 |
| `him_temperature` | 3.0，直到温度实验 |
| `sinkhorn_epsilon` | 0.05，直到 epsilon 实验 |
| `him_loss_coef` | 1.0，直到明确要求调整 |
| PPO/HIM clip | 保持当前分别配置的数值 |
| reward、command、terrain | 全部保持不变 |

E0/E1 已使用 `num_steps_per_env=100` 和 `save_interval=10` 完成。当前为了执行
E1-long，`him-collapse-step0` 分支临时改为 `num_steps_per_env=24` 和
`save_interval=100`；`num_envs`、seed 和训练轮数由命令行显式指定。E1-long
是验证训练成熟度的明确例外，不能把它与前 100 轮的单个 iteration 数值直接比较。

如果当前显存无法支持 `2048 × 100`，可以统一改为 `2048 × 24`，但必须从 E0 开始重新建立基线，不能直接拿不同 rollout 长度的实验结果做数值对比。

每个实验使用独立的 `run_name`，建议：

```text
collapse_e0_baseline
collapse_e1_student_only
collapse_e2_freeze_prototypes
collapse_e3_epsilon_003
collapse_e4_standard_ce
collapse_e5_temperature_1
collapse_e6_hio_before_ppo
```

训练命令模板：

```bash
python legged_gym/scripts/train.py \
    --task=go2 \
    --headless \
    --num_envs=2048 \
    --seed=1 \
    --max_iterations=100 \
    --run_name=collapse_e0_baseline
```

开始每次实验前检查：

```bash
git status --short
git diff --check
```

## 5. 坍缩判定指标

### 5.1 必须新增的训练指标

第一步只增加诊断，不改变训练计算。需要在 `estimator.py`、`ppo.py` 和 runner 日志中增加以下指标。

Prototype 几何指标：

- `Prototype effective rank`；
- `Prototype stable rank`；
- `Prototype mean abs off-diagonal cosine`；
- `Prototype same-direction fraction`，统计 cosine 大于 `0.99` 的非对角 pair；
- `Prototype collinear fraction`，统计绝对 cosine 大于 `0.99` 的非对角 pair。

Source/target latent 指标：

- `Source latent mean norm`；
- `Source latent centered effective rank`；
- `Source latent random-pair cosine mean`；
- target 对应的三项指标；
- `Positive source-target cosine`；
- 如果当前实验同时包含 teacher/student，应分别记录两组 source-target 指标。

Prototype 分配指标：

- 使用未经过 Sinkhorn 的 prototype score 做 `argmax`，记录 hard usage；
- `Hard prototype active count`；
- `Hard prototype perplexity`；
- 保留现有 soft usage，但将它明确标记为 `him_temperature=3.0` 下的 soft usage，不能将其当成实际激活数量。

梯度指标：

- clipping 前的 source encoder grad norm；
- clipping 前的 target encoder grad norm；
- clipping 前的 prototype grad norm；
- HIM 梯度超过 `him_max_grad_norm` 的更新比例。

注意：这里的 `Hard prototype perplexity` 是 prototype 使用分布的有效类别数，与 t-SNE 的 perplexity 完全不是同一个概念。

### 5.2 指标计算定义

Prototype effective rank 使用归一化后的 prototype 矩阵计算：

```python
singular_values = torch.linalg.svdvals(normalized_prototypes)
probabilities = singular_values.square()
probabilities = probabilities / probabilities.sum()
effective_rank = torch.exp(
    -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
)
```

Latent effective rank 必须先对 batch 维度中心化，再使用相同方法计算。不能直接对未中心化 latent 做 SVD，否则共同均值方向会掩盖真实坍缩。

Hard prototype perplexity 定义为：

```python
hard_index = scores.argmax(dim=-1)
usage = torch.bincount(hard_index, minlength=num_prototypes).float()
usage = usage / usage.sum()
hard_perplexity = torch.exp(
    -(usage * usage.clamp_min(1e-12).log()).sum()
)
```

### 5.3 暂定通过和失败标准

以下阈值是工程诊断阈值，不是论文给出的理论常数。所有实验使用同一阈值即可。

出现下列任一情况并持续 5 个 iteration，判定为坍缩：

- prototype effective rank 小于 `2.0`；
- prototype collinear fraction 大于 `0.90`；
- source latent random-pair cosine mean 大于 `0.98`，且 centered effective rank 小于 `2.0`；
- hard prototype perplexity 小于 `2.0`。

第 100 轮至少满足以下条件，才允许进入 500 轮验证：

- prototype effective rank 不低于 `4.0`；
- prototype collinear fraction 小于 `0.50`；
- source 和 target centered effective rank 均不低于 `3.0`；
- hard prototype perplexity 不低于 `4.0`；
- 最近 20 轮没有触发持续坍缩条件；
- velocity loss 保持有限，并且没有相对基线明显恶化；
- PPO reward、episode length 和地形等级没有异常下降。

### 5.4 100 轮体检与地形聚类不是同一个判据

100 轮实验只用于尽早排除明显坍缩，不用于宣称已经学到地形语义。训练初期
机器人主要处于站立、跌倒和动作探索阶段，即使 latent 保持多维，结构也更可能
反映姿态、速度和动作响应，而不是 stairs、slope 等地形标签。

因此需要严格区分：

- **100 轮抗坍缩体检**：只检查 prototype/latent 是否保持多个有效方向、hard
  assignment 是否没有集中到极少数 prototype，以及训练数值是否稳定；
- **500/1000 轮表示验证**：策略形成稳定步态且课程进入多个地形等级后，才检查
  地形标签在高维 latent 和 t-SNE 中是否形成可解释结构；
- **地形聚类失败**不能由第 100 轮 t-SNE 得出；反过来，如果第 100 轮已经出现
  effective rank 约为 1、random-pair cosine 接近 1 和单 prototype 占用接近 100%，
  则可以提前判定为表示坍缩。现有 1000 至 20000 轮 checkpoint 已证明这种坍缩
  不会仅靠增加训练轮数自行恢复。

HIM 的 prototypes 是自监督动力学原型，并不与四种人工 terrain label 一一对应。
最终可视化应在策略能稳定走过多种地形后采集，并同时报告 reward、terrain level
和各类样本数量，不能只看颜色是否分开。

## 6. 分步执行方案

### 步骤 0：补齐诊断和 checkpoint 轮数语义

目的：在不改变训练目标的情况下，让后续结论可以直接由日志和 checkpoint 验证。

当前实现状态（`him-collapse-step0` 分支）：

- 诊断指标、teacher/student 分组、梯度指标和离线分析脚本已实现；
- 新训练会保存真正初始化状态的 `model_0.pt`；
- `model_N.pt` 和 checkpoint 内的 `iter=N` 都表示已经完成 N 次更新；
- 静态、合成指标和 checkpoint 轮数测试已通过；
- Isaac Gym smoke test 和 E0 训练由用户在有 GPU 的 `feltio` 环境中执行。

预计修改：

- `rsl_rl/rsl_rl/modules/estimator.py`
  - 返回 prototype、source latent、target latent 和 hard assignment 诊断指标；
- `rsl_rl/rsl_rl/algorithms/ppo.py`
  - 聚合 HIM 指标；
  - 在 teacher/student 混合模式下，额外记录两组诊断；
  - 记录 clipping 前梯度；
- `rsl_rl/rsl_rl/runners/on_policy_runner.py`
  - 写入 TensorBoard；
  - 确保 `model_10.pt` 表示确实完成了第 10 轮，而 checkpoint 内 `iter` 也为 10；
- `legged_gym/scripts/analyze_him_checkpoint.py`
  - 增加不启动 Isaac Gym 的 checkpoint 几何分析脚本；
  - checkpoint 本身只直接分析 prototype；如果要分析 latent，必须额外传入之前采集的 latent 数据文件，不能从权重凭空计算 latent 分布。

要求：

- 不改变 loss、optimizer、采样比例和更新顺序；
- 诊断计算使用 `torch.no_grad()`；
- 如果完整 pairwise cosine 占用显存，只采样固定数量的 latent pair；
- 不把诊断 tensor 保存在 GPU list 中跨 iteration 累积。

验证命令：

```bash
conda activate feltio

python -m py_compile \
    rsl_rl/rsl_rl/modules/estimator.py \
    rsl_rl/rsl_rl/algorithms/ppo.py \
    rsl_rl/rsl_rl/runners/on_policy_runner.py \
    legged_gym/scripts/analyze_him_checkpoint.py
```

先使用少量环境运行 1 轮 smoke test：

```bash
python legged_gym/scripts/train.py \
    --task=go2 \
    --headless \
    --num_envs=64 \
    --seed=1 \
    --max_iterations=1 \
    --run_name=collapse_step0_smoke
```

该 run 必须同时生成 `model_0.pt` 和 `model_1.pt`，并且控制台能够打印 prototype effective rank、student latent effective rank、hard prototype perplexity 和 HIM grad norm。

查看 TensorBoard：

```bash
tensorboard --logdir logs/rough_go2_tshim2 --port 6006
```

smoke test 至少确认以下 tags 存在且数值有限：

```text
Representation/prototype_effective_rank
Representation/prototype_stable_rank
Representation/prototype_collinear_fraction
Representation/source_latent_centered_effective_rank
Representation/target_latent_centered_effective_rank
Representation/teacher_source_latent_centered_effective_rank
Representation/student_source_latent_centered_effective_rank
Representation/hard_prototype_active_count
Representation/hard_prototype_perplexity
Representation/source_encoder_grad_norm
Representation/target_encoder_grad_norm
Representation/prototype_grad_norm
Representation/him_grad_norm
Representation/him_grad_clipped_fraction
```

运行 E0：

```bash
python legged_gym/scripts/train.py \
    --task=go2 \
    --headless \
    --num_envs=2048 \
    --seed=1 \
    --max_iterations=100 \
    --run_name=collapse_e0_baseline
```

E0 的预期结果是复现坍缩。如果 E0 在相同配置下没有复现，需要先检查 seed、未提交配置和 checkpoint 加载状态，不进入步骤 1。

分析 checkpoint：

```bash
python legged_gym/scripts/analyze_him_checkpoint.py \
    --checkpoint_path=logs/rough_go2_tshim2/<E0目录>/model_100.pt
```

### 步骤 1：HIM 只使用 student 样本

这是第一个真正改变训练行为的实验，也是当前优先级最高的因果验证。

新增一个显式配置项，例如：

```python
him_sample_mode = "student"
```

允许值：

- `"all"`：原始行为，用于 E0；
- `"student"`：当前诊断分支行为，只对 student 部分执行 velocity 和 SwAV representation update，用于 E1。

在 `ppo.py` 的 representation update 中，根据 `teacher_samples` 切分：

```python
representation_start = teacher_samples if self.him_sample_mode == "student" else 0
representation_end = teacher_samples + student_samples

representation_history = history_batch[representation_start:representation_end]
representation_velocity_target = velocity_target_batch[
    representation_start:representation_end
]
representation_next_response = next_response_batch[
    representation_start:representation_end
]
representation_valid_him_target = valid_him_target_batch[
    representation_start:representation_end
]
```

要求：

- PPO actor/critic mini-batch 仍然保持 75% teacher、25% student，不改 PPO 训练；
- student-only 应同时作用于 HIM velocity loss 和 SwAV loss；
- 不改 `him_temperature`、epsilon、loss reduction、prototype 更新和优化器；
- 不从 E0 checkpoint 恢复，重新随机初始化 E1。
- 2048 环境、100 步 rollout、4 个 mini-batch 时，控制台中的 `HIM samples/update` 应为 `12800`，而不是 E0 的 `51200`。
- E1 中不带前缀的 source/target/hard prototype 指标应基于 student slice；`teacher_*` 和 `student_*` 指标仍同时保留用于旁路比较。

运行 E1：

```bash
python legged_gym/scripts/train.py \
    --task=go2 \
    --headless \
    --num_envs=2048 \
    --seed=1 \
    --max_iterations=100 \
    --run_name=collapse_e1_student_only
```

决策：

- 如果 E1 通过第 100 轮标准，说明 teacher 样本混入 HIM 是主要原因。保留 student-only，跳过步骤 2 至步骤 5，先进入步骤 6 检查更新顺序，再做 500/1000 轮验证；
- E1 实际在第 100 轮仍然坍缩；考虑到当时尚无稳定步态，先进入步骤 1B，而不是立即进入 E2；
- 如果 E1 的表示不坍缩但 PPO 明显退化，先检查 student batch 大小和 velocity loss，不能立即恢复 teacher HIM 样本。

### 步骤 1B：E1-long，验证训练成熟后是否自行恢复

目的：直接检验“100 轮时机器人还不会走，所以聚类不明确”这一解释。该实验
仍是 E1 的 student-only 训练，不加入任何 E2 prototype freeze。

唯一相对 E1 短实验的运行条件变化：

- `num_steps_per_env`：`100 -> 24`；
- `max_iterations`：`100 -> 10000`；
- `save_interval`：`10 -> 100`，避免生成 1000 个大 checkpoint。

保持不变：

- `num_envs=2048`，不因为换到 4090 就同时增加环境数；
- `seed=1`；
- `him_sample_mode=student`；
- `freeze_prototype_updates=0`；
- teacher ratio、loss、temperature、Sinkhorn、优化器和 reward/terrain 配置。

当前每轮仍有 `5 × 4 = 20` 次 HIM optimizer update，但每次 student HIM batch
从 `12800` 变为 `3072`。控制台必须显示：

```text
HIM samples/update: 3072
Prototype frozen: 0.00%
```

运行命令：

```bash
conda activate feltio

python legged_gym/scripts/train.py \
    --task=go2 \
    --headless \
    --num_envs=2048 \
    --seed=1 \
    --max_iterations=10000 \
    --run_name=collapse_e1_long_rollout24_10000
```

必须从随机初始化开始，不设置 `--resume`。旧 E1 每个环境经历了
`100 iterations × 100 steps = 10000 steps`，在新实验中约对应
`10000 / 24 = 416.7` 轮，因此优先比较：

| 新实验 checkpoint | 每环境累计步数 | 用途 |
| --- | ---: | --- |
| `model_100.pt` | 2400 | 观察早期坍缩速度 |
| `model_400.pt` | 9600 | 与旧 E1 第 100 轮近似等数据量比较 |
| `model_1000.pt` | 24000 | 检查步态形成后的首次恢复迹象 |
| `model_2000.pt` | 48000 | 检查课程和表示趋势 |
| `model_5000.pt` | 120000 | 检查中期稳定性 |
| `model_10000.pt` | 240000 | 最终结论 |

判读时把表示指标与 reward、episode length、student/teacher terrain level 放在
同一时间轴：

- 如果步态和 terrain level 已明显提升，但 prototype rank 仍约为 1、latent
  random-pair cosine 仍接近 1，且 hard usage 长期集中，则排除“只是轮数不够”，
  再进入 E2；
- 如果高维表示在步态形成后从坍缩状态持续恢复，并且多个 checkpoint 保持健康，
  则训练成熟度确实是主要因素，先不要执行 E2；
- 如果 10000 轮仍没有稳定步态或 terrain level 始终为 0，本实验不能回答地形
  聚类问题，应先定位 PPO/课程学习，而不能把颜色未分开归因于 HIM；
- t-SNE 只在已经形成稳定步态且覆盖多个地形等级的 checkpoint 上绘制。

### 步骤 2：训练初期冻结 prototypes

只有 E1-long 在策略成熟后仍然坍缩时才执行本步骤；当前暂停。

目的：避免训练刚开始时随机 encoder 输出和 prototype 同时快速追逐，直接进入共线局部解。

新增配置：

```python
freeze_prototype_updates = 300
```

这里按 HIM optimizer step 计数，不按 PPO iteration 计数。当前每轮约有 `num_learning_epochs × num_mini_batches` 次 HIM update，因此 300 次通常覆盖最初约 15 轮。

实现要求：

- freeze 期间 encoder 和 velocity head 正常更新；
- 只阻止 `prototypes.weight` 更新；
- 达到 update 数量后自动恢复 prototype 梯度；
- 日志记录 `Prototype frozen` 和累计 update 数；
- 其余配置与 E1 完全相同。

当前实现状态（`him-collapse-step0` 分支，已关闭、未运行）：

- prototype freeze 诊断开关已经实现，但当前实验配置明确设置为
  `freeze_prototype_updates=0`；
- `ppo.py` 在反向传播后先记录 prototype 原始梯度，再把冻结阶段的
  `prototypes.weight.grad` 置为 `None`，因此只阻止 prototype optimizer step；
- `him_update_count` 按实际 HIM mini-batch 更新累计；
- checkpoint 保存并恢复 `him_update_count`，旧 checkpoint 缺少该字段时按
  `iter × num_learning_epochs × num_mini_batches` 推算，避免恢复训练后重新冻结；
- TensorBoard 自动记录 `prototype_frozen_fraction` 和 `him_update_count`，控制台
  同时打印 `Prototype frozen` 与 `HIM update count`。

当前每轮有 `5 × 4 = 20` 次 HIM update，因此新训练应满足：

| checkpoint/iteration | `him_update_count` | prototype 状态 |
| --- | ---: | --- |
| `model_0.pt` | 0 | 初始化状态 |
| iteration 10 / `model_10.pt` | 200 | 全部冻结 |
| iteration 15 | 300 | 冻结阶段结束 |
| iteration 20 / `model_20.pt` | 400 | 已正常更新 100 次 |
| iteration 100 / `model_100.pt` | 2000 | 已正常更新 1700 次 |

iteration 1 至 15 的 `prototype_frozen_fraction` 应为 `1.0`，从 iteration 16
开始应为 `0.0`。离线分析时，`model_0.pt` 与 `model_10.pt` 的归一化 prototype
几何指标应相同；若不同，说明冻结没有生效。

运行 E2：

```bash
python legged_gym/scripts/train.py \
    --task=go2 \
    --headless \
    --num_envs=2048 \
    --seed=1 \
    --max_iterations=100 \
    --run_name=collapse_e2_freeze_prototypes
```

必须从随机初始化开始，不从 E0/E1 checkpoint 恢复。训练完成后至少分析
`model_0.pt`、`model_10.pt`、`model_20.pt`、`model_50.pt` 和
`model_100.pt`。

决策：

- E2 通过第 100 轮抗坍缩体检：说明它是可继续验证的候选配置，不代表已经实现
  地形聚类；保留 student-only 和 prototype freeze，跳到步骤 6，再进入 500/1000
  轮表示验证；
- `model_0` 与 `model_10` 的 prototype 几何不同：冻结实现有误，停止实验并修复；
- 冻结阶段正常，但在 iteration 16 解冻后很快坍缩：说明固定 prototype 只能延迟
  坍缩。优先增加一个独立的“延迟启动 SwAV”实验，验证早期探索数据缺乏运动和
  地形多样性是否是原因；
- E2 到第 100 轮仍未出现 terrain label 分离、但高维抗坍缩指标健康：不能判为
  失败，应继续到稳定步态和多个地形等级后再评估；
- E2 在冻结期间 encoder/latent 已经坍缩：prototype 共适应不是充分解释，记录
  结果后再进入步骤 3。

延迟启动 SwAV 是 E2 数据触发的候选实验，本步骤暂不实现。若需要执行，应保持
velocity estimation 正常训练，只把 SwAV loss 的启用时间延后到基线首次形成稳定
步态附近；具体 warm-up 轮数必须依据 reward、episode length 和 terrain level
确定，不能因为 100 是整百数就直接选 100。

### 步骤 3：把 Sinkhorn epsilon 从 0.05 改为 0.03

只有 E2 仍然坍缩时才执行本步骤。

修改：

```python
sinkhorn_epsilon = 0.03
```

其余配置与 E2 相同。不要同时修改 Sinkhorn iteration 数量或 `him_temperature`。

运行 E3：

```bash
python legged_gym/scripts/train.py \
    --task=go2 \
    --headless \
    --num_envs=2048 \
    --seed=1 \
    --max_iterations=100 \
    --run_name=collapse_e3_epsilon_003
```

决策：

- E3 通过：保留当前设置，跳到步骤 6；
- E3 失败：进入步骤 4。

### 步骤 4：把 SwAV loss 改为标准 soft cross-entropy reduction

当前实现对 batch 和 prototype 两个维度一起 `.mean()`，会使 SwAV loss 和相应梯度额外缩小约 `num_prototypes` 倍。标准写法是先对 prototype 维度求和，再对 batch 求均值：

```python
history_cross_entropy = -(
    target_assignments * history_log_prob
).sum(dim=-1).mean()

target_cross_entropy = -(
    history_assignments * target_log_prob
).sum(dim=-1).mean()

swav_loss = 0.5 * (history_cross_entropy + target_cross_entropy)
```

要求：

- 本步骤保持 `him_loss_coef=1.0`；
- 保持 `him_max_grad_norm` 当前值；
- 重点观察 clipping 前梯度和 clipping 比例；
- 不要为了让日志数值接近旧值，再除以 `num_prototypes`，否则等同于恢复旧 reduction；
- 其余设置与 E3 相同。

运行 E4：

```bash
python legged_gym/scripts/train.py \
    --task=go2 \
    --headless \
    --num_envs=2048 \
    --seed=1 \
    --max_iterations=100 \
    --run_name=collapse_e4_standard_ce
```

决策：

- E4 表示恢复且 PPO 正常：保留标准 reduction，进入步骤 6；
- E4 不坍缩但 HIM 几乎每步都被裁剪：保持 reduction，单独增加一个 HIM loss coefficient 实验，例如 `1/16`，不要与其他参数一起改；
- E4 仍坍缩：进入步骤 5。

### 步骤 5：单独降低 temperature

只有 E4 仍然坍缩时才执行本步骤。

第一次只把：

```python
him_temperature = 1.0
```

不要直接同时改成多个值。运行 E5 后，如果仍失败，再依次尝试 `0.5` 和 `0.1`，每个值都建立独立 run。

运行 E5：

```bash
python legged_gym/scripts/train.py \
    --task=go2 \
    --headless \
    --num_envs=2048 \
    --seed=1 \
    --max_iterations=100 \
    --run_name=collapse_e5_temperature_1
```

判断时以 hard usage 和高维 effective rank 为准。temperature 变低会天然让 soft usage 看起来更尖锐，所以不能只比较 soft usage min/max。

如果尝试到 `temperature=0.1` 仍然坍缩，不继续盲调 temperature，进入“步骤 8：保底正则实验”。

### 步骤 6：在表示稳定后检查 HIO/PPO 更新顺序

本步骤不是首要修复项。只有前面某个实验已经让表示通过 100 轮标准后才执行。

当前实现是：

```text
完整 PPO update -> 完整 HIM update
```

论文流程更接近：

```text
HIO representation update -> PPO update
```

发布仓库则是在每个 mini-batch 内先更新 estimator，再更新 PPO。为了控制变量，本项目先只比较完整两阶段的顺序，不立即重写成 interleaved。

E6 只改变顺序为：

```text
完整 HIM update -> 完整 PPO update
```

要求：

- mini-batch 内容和数量不变；
- HIM 仍使用已经确定的 sample mode；
- 不改 optimizer 和 learning rate；
- 不使用任何前序实验 checkpoint 恢复，重新初始化；
- 比较表示指标之外，还要比较 PPO reward 和 KL。

运行 E6：

```bash
python legged_gym/scripts/train.py \
    --task=go2 \
    --headless \
    --num_envs=2048 \
    --seed=1 \
    --max_iterations=100 \
    --run_name=collapse_e6_hio_before_ppo
```

决策：

- 两种顺序都稳定：选择 reward、速度估计和训练耗时更好的实现；
- 只有 HIO-first 稳定：采用 HIO-first；
- 只有当前 PPO-first 稳定：保留当前顺序，并在文档中记录它与论文流程的差异；
- 不在同一实验里继续改成 interleaved。

### 步骤 7：扩大到 500 和 1000 轮

确定最终短实验配置后，先用相同配置补做 seed 2 和 seed 3 的 100 轮短实验。三个 seed 都不能触发持续坍缩条件；如果只有部分 seed 通过，当前改动还不能判定为稳定修复，应继续定位初始化敏感性。

补充 seed 命令模板：

```bash
python legged_gym/scripts/train.py \
    --task=go2 \
    --headless \
    --num_envs=2048 \
    --seed=2 \
    --max_iterations=100 \
    --run_name=collapse_final_seed2_100

python legged_gym/scripts/train.py \
    --task=go2 \
    --headless \
    --num_envs=2048 \
    --seed=3 \
    --max_iterations=100 \
    --run_name=collapse_final_seed3_100
```

三个 seed 都通过后，重新从随机初始化运行，不从任何坍缩 checkpoint 恢复。

先运行 500 轮：

```bash
python legged_gym/scripts/train.py \
    --task=go2 \
    --headless \
    --num_envs=2048 \
    --seed=1 \
    --max_iterations=500 \
    --run_name=collapse_final_500
```

检查 `model_100.pt`、`model_200.pt`、`model_300.pt`、`model_400.pt` 和 `model_500.pt`。全部满足高维标准后，再运行 1000 轮：

```bash
python legged_gym/scripts/train.py \
    --task=go2 \
    --headless \
    --num_envs=2048 \
    --seed=1 \
    --max_iterations=1000 \
    --run_name=collapse_final_1000
```

1000 轮通过后，才把同样配置迁移到 4090 做长训练。迁移显卡时可以增加环境数量，但先保持其他超参数不变，重新建立一个 100 轮显存和数值稳定性基线。

### 步骤 8：保底正则实验

仅当 student-only、prototype freeze、epsilon、标准 CE 和 temperature 实验都不能避免坍缩时执行。

此阶段已经开始偏离原始 HIMLoco，所有正则必须作为单独 ablation：

1. prototype repulsion：惩罚归一化 prototype 的非对角 cosine 平方；
2. latent variance regularization：要求每个 latent 维度保持最小标准差；
3. latent covariance regularization：惩罚中心化 latent covariance 的非对角项。

推荐顺序是先 prototype repulsion，再 variance，最后 covariance。每次只加入一种，并记录其 coefficient。不能把三种正则一次性全部加入。

## 7. 实验决策树

```text
E0 当前实现能否复现坍缩？
├── 否：先检查实验配置和 checkpoint 状态，不继续
└── 是：E1 只用 student 样本训练 HIM
    ├── 通过：teacher 样本混入是主要原因 -> E6 更新顺序实验
    └── 失败：E2 增加 prototype freeze
        ├── 通过：保留 freeze -> E6
        └── 失败：E3 epsilon=0.03
            ├── 通过：保留 epsilon -> E6
            └── 失败：E4 标准 soft cross-entropy
                ├── 通过：保留标准 CE -> E6
                └── 失败：E5 temperature=1.0/0.5/0.1
                    ├── 通过：E6
                    └── 失败：E8 单项 anti-collapse 正则
```

如果某一步通过，先跳到 E6，不要为了“可能更好”继续把后面的所有改动叠加上去。最终应采用能够稳定训练的最小改动集合。

## 8. 最终聚类验证

### 8.1 先验证高维空间

每个目标 checkpoint 都需要输出：

- prototype effective rank、stable rank 和 cosine matrix；
- source/target centered effective rank；
- hard prototype usage 和 hard perplexity；
- 正样本 cosine 与随机负样本 cosine；
- terrain label 的 silhouette score；
- 一个简单 linear probe 或 k-NN terrain accuracy。

terrain silhouette 和 probe 只作为后验分析，因为 HIM 并没有使用 terrain label 做监督。即使 representation 健康，也不能要求每一种人工地形标签都形成完全分离的团块。

### 8.2 再生成 t-SNE

t-SNE 只用于展示，不用于判断是否坍缩。最终可视化需要固定：

- 相同 checkpoint；
- 相同 latent 数量和抽样规则；
- 相同 terrain category 比例；
- 相同机器人 command；
- 相同标准化方式；
- 相同 t-SNE seed、perplexity 和迭代次数。

为了和 HIMLoco 图更接近，最终评估应覆盖并平衡以下类别：

- stairs；
- rough slope；
- slope；
- discrete terrain。

需要保证不同类别的运动命令和地形难度尽量一致，否则 t-SNE 可能主要按速度、转向、地形等级或时间连续性分组，而不是按地形类型分组。

## 9. 每一步的记录模板

每完成一个实验，在本文件末尾复制并填写以下记录：

```text
实验编号：E?
Git commit：
Run 目录：
Seed：
num_envs：
num_steps_per_env：
唯一修改项：

Iteration 10：
  prototype effective rank：
  source centered effective rank：
  hard prototype perplexity：

Iteration 50：
  prototype effective rank：
  source centered effective rank：
  hard prototype perplexity：

Iteration 100：
  prototype effective rank：
  prototype collinear fraction：
  source centered effective rank：
  target centered effective rank：
  hard prototype perplexity：
  velocity loss：
  mean reward：
  terrain level：

结论：通过 / 坍缩 / PPO 退化 / 运行失败
下一步：
```

## 10. 提交策略

每个行为改动单独提交，不把多个实验合并到一个 commit：

```text
chore: add HIM collapse diagnostics
fix: train HIM representation on student samples
fix: freeze HIM prototypes during warmup
fix: use standard SwAV cross entropy reduction
experiment: adjust HIM Sinkhorn epsilon
experiment: adjust HIM temperature
experiment: update HIM before PPO
```

每个 commit body 分条记录：

- 修改的文件或组件；
- 改变的训练行为；
- 执行过的验证命令；
- 当前实验结果或尚未完成的训练验证。

## 11. 禁止事项

在定位完成前不要执行以下操作：

- 从已经坍缩的 checkpoint 继续训练来验证 anti-collapse 修改；
- 一次同时修改 sample mode、temperature、epsilon 和 loss reduction；
- 只依据 t-SNE 图片决定是否成功；
- 为了让 terrain 聚类更明显而修改训练 terrain label 或增加有监督分类 loss；
- 在不同 seed、rollout 长度、环境数量之间直接比较单个 loss 数值；
- 把 soft prototype usage 当成 hard prototype 激活数量；
- 在短实验还未通过前直接开始 1000 轮或更长训练。

## 12. 实验记录

### E0：teacher/student 混合样本基线

```text
实验编号：E0
Run 目录：logs/rough_go2_tshim2/Aug09_20-54-21_collapse_e0_baseline
Seed：1
num_envs：2048
num_steps_per_env：100
唯一实验条件：him_sample_mode 等价于 all

Model 0：
  checkpoint iter：0
  prototype effective rank：9.535460

Iteration 10：
  prototype effective rank：4.457013
  source centered effective rank：2.4198
  hard prototype perplexity：8.3290

Iteration 50：
  prototype effective rank：1.222270
  prototype collinear fraction：0.758333
  source latent mean norm：0.9992
  source random-pair cosine：0.9982
  hard prototype perplexity：1.5003

Iteration 100：
  checkpoint iter：100
  prototype effective rank：1.033542
  prototype stable rank：1.004663
  prototype mean absolute off-diagonal cosine：0.995018
  prototype collinear fraction：0.825000
  source centered effective rank：1.5994
  target centered effective rank：1.2400
  source latent mean norm：0.9962
  source random-pair cosine：0.9912
  hard prototype perplexity：2.2990
  velocity loss：0.037626
  HIM gradient clipping fraction：0.0000

结论：成功复现 prototype 和 latent 坍缩
下一步：E1，只把 HIM representation update 改为 student-only
```

### E1：HIM 只使用 student 样本

```text
实验编号：E1
Run 目录：logs/rough_go2_tshim2/Aug09_21-22-10_collapse_e1_student_only
Seed：1
num_envs：2048
num_steps_per_env：100
唯一修改项：him_sample_mode 从 all 改为 student
HIM samples/update：12800

Model 0：
  checkpoint iter：0
  prototype effective rank：9.535460

Iteration 10：
  prototype effective rank：4.550604
  source centered effective rank：2.4650
  source latent mean norm：0.5671
  source random-pair cosine：0.3209
  hard prototype perplexity：7.5335

Iteration 20：
  prototype effective rank：1.999411
  source centered effective rank：2.0368
  source latent mean norm：0.7727
  source random-pair cosine：0.5956
  hard prototype perplexity：5.6468

Iteration 50：
  prototype effective rank：1.519618
  prototype collinear fraction：0.758333
  source latent mean norm：0.9785
  source random-pair cosine：0.9581
  hard prototype active count：5
  hard prototype perplexity：2.4661

Iteration 100：
  checkpoint iter：100
  prototype effective rank：1.055065
  prototype stable rank：1.008908
  prototype mean absolute off-diagonal cosine：0.990400
  prototype collinear fraction：0.825000
  source centered effective rank：4.3621
  target centered effective rank：1.4640
  source latent mean norm：0.9995
  source random-pair cosine：0.9991
  positive source-target cosine：0.9990
  hard prototype active count：5
  hard prototype perplexity：1.1506
  hard prototype maximum usage：0.9717
  soft prototype perplexity：15.2287
  velocity loss：0.058751
  source encoder grad norm：0.250309
  target encoder grad norm：0.000014
  prototype grad norm：0.002786
  HIM gradient clipping fraction：0.0000
  student reward：-0.208915
  teacher reward：-0.176949

补充观察：
  student 和 teacher source random-pair cosine 分别为 0.9991 和 0.9979
  E0/E1 最终 prototypes 都形成相同的 4/12 正反方向结构
  student-only 只轻微延缓中期坍缩，没有改变最终吸引子

结论：坍缩；teacher 样本混入不是主要根因
下一步：E2，保留 student-only，只增加前 300 次 update 的 prototype freeze
```

### E1-long：student-only + rollout 24 + 10000 轮

```text
实验编号：E1-long
状态：待在 4090 上从随机初始化运行
目标 Run 名：collapse_e1_long_rollout24_10000
Seed：1
num_envs：2048
num_steps_per_env：24
max_iterations：10000
save_interval：100
him_sample_mode：student
freeze_prototype_updates：0
预期 HIM samples/update：3072

主要 checkpoint：model_100/400/1000/2000/5000/10000.pt
主要问题：表示会不会在稳定步态和多地形课程出现后自行恢复
结论：等待训练数据
```

### E2：student-only + 训练初期冻结 prototypes（暂停）

```text
实验编号：E2
状态：暂停，等待 E1-long 结论
Seed：1
num_envs：2048
num_steps_per_env：100
继承 E1：him_sample_mode=student
唯一修改项：freeze_prototype_updates 从 0 改为 300
预期 HIM samples/update：12800
预期冻结范围：iteration 1 至 15
预期恢复更新：iteration 16 起
目标 Run 名：collapse_e2_freeze_prototypes

必须记录：
  model_0/model_10 的 prototype 几何是否一致
  iteration 10/20/50/100 的 prototype effective rank
  iteration 10/20/50/100 的 source/target centered effective rank
  iteration 10/20/50/100 的 latent random-pair cosine
  iteration 10/20/50/100 的 hard prototype perplexity 和 maximum usage
  prototype frozen fraction、HIM update count 和三组梯度 norm
  velocity loss、teacher/student reward 与 terrain level

结论：等待 E2 训练数据
```
