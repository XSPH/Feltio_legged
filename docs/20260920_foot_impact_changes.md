# 2026-09-20 上楼落脚冲击改动记录

## 目的与依据

降低上楼时后脚猛烈落地和触地后的短时重压，同时保留正常支撑和上楼能力。
本次对四只脚使用相同规则，避免把冲击转移到前脚。

参考日志：`logs/mujoco/20260920_215708/`。异常段右后脚触地前向下速度约
2.86 m/s，竖直接触力峰值约 480 N、合力峰值约 565 N；一次连续超过
200 N 的竖直载荷持续约 70 ms。异常段平均竖直力约 41 N，问题主要表现为短时冲击。
这些数据来自 MuJoCo，不能直接当作 PhysX 或真机的安全限值。

## 已确认的改动范围

- `legged_gym/envs/base/legged_robot.py`：在物理子步记录触地前速度及接触力；修正
  `foot_impact_velocity`；新增触地后短窗口内的 `foot_impact_force`；处理接触去抖和环境重置。
- `legged_gym/envs/base/legged_robot_config.py`：增加冲击参数和默认关闭的新奖励入口。
- `legged_gym/envs/go2/go2_config.py`：启用新奖励并设置下表参数。
- `legged_gym/tests/test_foot_impact.py`：覆盖短时冲击、触地前速度、窗口、去抖、时间缩放和重置。

| 配置项 | 改动前 | 本次初值 |
| --- | --- | --- |
| `impact_speed_threshold` | 1.5 m/s | 1.2 m/s |
| `scales.foot_impact_velocity` | -0.05 | -0.05 |
| `impact_force_threshold` | 无 | 250 N |
| `impact_window_s` | 无 | 0.08 s |
| `impact_contact_gap_s` | 无 | 0.02 s |
| `scales.foot_impact_force` | 无 | -0.2（基础配置为 0） |

250 N 是软惩罚阈值，不是力矩或接触力硬限幅。上述权重和阈值是首轮训练初值，
需要结合成功上楼时的冲击分布评估。

## 统计与奖励定义

- 每个物理步读取足端世界坐标系竖直速度和竖直接触力；当前配置为 5 ms/步。
- 足端连续失去竖直支撑达到 20 ms 后，重新获得支撑算一次新触地。
  更短的接触抖动不重复结算速度惩罚，也不延长冲击窗口。
- 保存触地前最后一个无支撑物理步的速度，按
  `max(-v_z_before_contact - impact_speed_threshold, 0)^2` 结算一次速度代价。
  速度代价沿用原有事件奖励的控制周期缩放。
- 新触地开启 80 ms 窗口，含触地所在物理步；窗口按仿真时间连续递减。
  窗口内累计 `max(F_z / impact_force_threshold - 1, 0)^2`。
- 力代价先在一个控制周期内对子步取平均，再交由现有奖励框架乘控制周期时长，
  等价于一次时间积分，避免重复乘时间或因 decimation 重复放大。
- 每个控制周期清空代价累计值，跨周期保留窗口和接触历史；环境重置时清空该环境全部冲击状态。
- 原有步态/腾空时间奖励的接触状态仍按原逻辑更新。

## 接口与工作区

本次只改变训练奖励及统计，观测顺序、45 维本体观测、5 帧历史、12 维动作、
动作缩放、归一化、PD 和导出接口保持一致。新行为需微调/训练并重新导出模型后生效。

开始时工作分支为 `cts`，基准提交为 `3fab56f`，已有用户修改：`deploy/configs/config.yaml`、
`deploy/models/policy.mnn`、`legged_gym/scripts/play.py`。本次保留这些修改。

## 验证记录

- 已通过：新增 10 项冲击统计测试及现有 4 项地形测试，共 14 项。测试使用 CPU 张量和
  模拟的 Gym 接口，不创建物理仿真。覆盖触地前速度保留、5 ms 脉冲在控制周期结束前消失、
  80 ms 窗口到期、20 ms 去抖、四脚规则一致、环境局部重置，以及通过真实奖励结算函数
  验证不同物理步长/decimation 下力代价只积分一次。
- 已通过：四个新增/修改 Python 文件的 `py_compile` 语法检查；`git diff --check`。
- 用户提出“不用执行仿真测试”前已启动一次 CPU PhysX 冒烟检查；后续收取到完成结果：
  4 个平地环境运行 200 个控制步（800 个物理步），观测及奖励均为有限值，重置检查通过。
  该检查不代表爬楼或冲击改善验证。收到该要求后未再启动或重跑仿真，确认没有遗留仿真进程。
- 开启任一冲击奖励后，需要每个物理步刷新接触力和刚体状态；GPU 训练吞吐尚未测量。
- 尚未训练新策略。部署效果需在相同楼梯上比较后脚触地前速度、竖直力 P95/峰值、
  超阈值载荷累计量、上楼成功率和卡滞时长，并覆盖低速与满速度。

非仿真单元测试复现命令（使用 `feltio` 环境，扩展构建和绘图缓存放在 `/tmp`）：

```bash
PATH=/home/asuka/miniconda3/envs/feltio/bin:$PATH \
TORCH_EXTENSIONS_DIR=/tmp/feltio-impact-torch-extensions \
MPLCONFIGDIR=/tmp/feltio-impact-mpl \
/home/asuka/miniconda3/envs/feltio/bin/python -m unittest discover -s legged_gym/tests -v
```
