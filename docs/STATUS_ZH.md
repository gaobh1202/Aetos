# 项目现状与迁移验证清单

整理日期：2026-11-15。GitHub 版本：`gaobh1202/Aetos` main = `764ed5b`；
OmniDrones fork：`gaobh1202/amov_omni` main = `6532697`。
开发环境：Isaac Sim 4.1.0（`omni.isaac.core` 命名空间）、Python 3.10、单机 31 GB 内存 / 16 GB 显存。

标注"未复核"的条目是早期完成、这次整理时没有重新跑过的。

## 1. 已有内容（GitHub 上都有）

### 1.1 机型与资产
- 六个预设机型全部可生成、可飞：`iris650_soarm100`、`amov950_airbot_bottom`、
  `amov950_airbot_dexterous_bottom`、`uaquad600_gripper`、`fahexa750_gripper`、
  `hummingbird340_2dof_gripper`。
- 运行时组合（无人机 × 机械臂 × 夹爪），也可导出为单个 USD（`scripts/export_robot_usd.py`）。
- 所有外部资产路径已收进仓库目录；二进制资产不在 git 里，见第 3 节。

### 1.2 无人机控制
- 位置控制器（Lee）、全驱动六旋翼的 6 自由度力矩分配。
- 总推力 + 机体角速率（CTBR，对应 PX4 offboard rate 模式）动作接口和角速率内环
  （`controllers/drone/px4_rate_controller.py`）。
- 机械臂反作用隔离：`arm_dynamics_residual`、`arm_locked_joints`、`arm_write` 按机型配置。
- 键盘遥操作（无人机和机械臂分开选设备）、SO-ARM 主手遥操作。

### 1.3 机械臂控制与规划
- 通用末端笛卡尔控制器（差分 IK，阻尼最小二乘，`position_first` 严格优先级模式），
  六个机型都接入：`scripts/arm_mocap_control.py`。
- SO101 的 IK、cuRobo 无碰撞抓取规划（`scripts/arm_planning/`，在 `armplan` 环境运行，不依赖 Isaac）。
- 可行基座集合：`scripts/arm_planning/feasible_base_set.py`，按桨叶间隙、仰角、可达性、
  关节限位余量过滤；5000 个采样得到 793 个可行位姿，结果在
  `aetos_multiam/data/aerial_transport/so101_feasible_bases.npz`。

### 1.4 任务环境
- 单机：`AerialBucketGrasp`、`AerialButtonGrasp`、`ButtonGrasp`。
- 多机：`AerialTransport`、`AerialTransportTrack`、`AerialAssembly`、`AerialAssemblyTriple`、
  `AerialHandover`、`AerialCabinetGrasp`。
- `AerialTransport` 有脚本化演示采集（仓库里带 10 个回合）；`AerialTransportTrack` 是跟踪这些
  演示的 RL 环境，SAC 在 32 个并行环境下能训练（未复核）。

### 1.5 机械臂抓取数据与视觉运动策略（plan.md 阶段 2）
- 采集：`scripts/collect_arm_grasp_demo.py`。无人机固定在一个会漂移的运动学支架上
  （模拟悬停误差），臂从默认姿态出发抓箱子把手并抬起，所有接触真实开启。
- 导出 LeRobot 数据集：`scripts/export_arm_grasp_lerobot.py`（文件头有 ACT 训练命令）。
- 闭环评测：`scripts/serve_lerobot_policy.py`（lerobot 环境）+
  `collect_arm_grasp_demo.py --policy`（Isaac 环境）。

## 2. 只在本机、GitHub 上没有的内容

| 内容 | 状态 |
|---|---|
| `scripts/base_state.py`，以及采集 / 导出 / 服务三个脚本的后续改动（`--base-state`、`--n-action-steps`、`--temporal-ensemble`） | 已验证可用，未提交 |
| `AerialGraspAssist` 环境、配置、检查脚本（无人机顺应机械臂舒适度） | 仍在修改，按要求暂不提交 |
| `aetos_multiam/data/aerial_transport/arm_grasp_plans.npz`（282 条抓取规划） | 未跟踪；`exec_so101_grasp_isaac.py` 的默认 `--plans` 指向它 |
| `plan.md`、`README_ZH.md` 的改动、`examples/show_scene_default.png` | 暂不提交 |
| 原始演示 `outputs/arm_grasp_demos`（249 回合，5.7 GB） | 本机 |
| LeRobot 数据集（252 MB；带基座状态的另一份同样大小） | 第一份已打包：`/home/benhua/aetos_so101_handle_grasp_lerobot_20261114.tar.gz` |
| ACT 检查点 `outputs/train/act_so101_handle_grasp`、`..._base`（各 2.4 GB） | 本机 |

## 3. 另一台主机需要的非 git 内容

- 资产压缩包：`aetos_multiam_meshes_20261114_v2.tar.gz`（132 MB，取代已上传的 `_20261114` 版）
  和 `aetos_multiam_object_assets_20261114.tar.gz`（227 MB），解压位置见 `docs/ASSETS.md`。
- OmniDrones：直接检出 fork 的 `6532697`，或在 `f98c187` 上运行 `third_party/omnidrones/apply.sh`。
- 三个 conda 环境：`omnidrone`（Isaac）、`armplan`（mujoco、placo、cuRobo）、
  `lerobot`（lerobot 0.4.4；torchcodec 加载失败时用 `--dataset.video_backend=pyav`）。

## 4. 迁移验证清单

由浅到深；前一级不过，后面的不用试。期望值是在 Isaac Sim 4.1 上测得的。

### 第 0 级：不依赖 Isaac
```bash
conda activate armplan
python scripts/arm_planning/check_arm_kinematics.py
python scripts/arm_planning/check_so101_kinematics.py
python scripts/arm_planning/feasible_base_set.py --num 200 --out /tmp/bases.npz
```
期望：自检通过；200 个采样里约 16% 可行。

### 第 1 级：资产加载与渲染
```bash
python examples/show_robot.py --robot all --headless
python examples/show_scene.py --scene default --headless
```
期望：六个机型都出现在截图里，机械臂装在正确位置，无穿模。

### 第 2 级：关节驱动与雅可比
```bash
python scripts/arm_mocap_control.py --preset iris650_soarm100 --test
```
六个预设都跑一遍。期望：目标保持 1 秒后的末端位置误差 0.3–0.5 mm
（`hummingbird340_2dof_gripper` 约 0.1 mm）；`amov950_airbot_*` 的姿态误差 ≤ 0.3°。

### 第 3 级：飞行
```bash
python debug/verify_iris650_flight.py
python examples/show_am_action.py --preset all --headless
```
期望：六个机型都完成起飞、平移、偏航、摆臂、降落；机身倾角量级见 README。

### 第 4 级：接触与抓取
```bash
python scripts/arm_planning/exec_so101_grasp_isaac.py --drive --plans aetos_multiam/data/aerial_transport/so101_feasible_bases.npz
python scripts/collect_arm_grasp_demo.py --episodes 20 --no-cameras --jitter 0 --action-noise 0
python scripts/collect_arm_grasp_demo.py --episodes 50
```
期望：规划回放 789/793；无扰动抓取 19/20；默认扰动 42/50（约 84%）。最后一条同时检验相机渲染。

### 第 5 级：任务环境与 RL
```bash
python scripts/run_collab_smoke.py task=AerialTransport headless=true task.env.num_envs=2
python scripts/collect_aerial_transport_demo.py --num-episodes 2
python debug/check_aerial_transport_track.py --num-envs 16
python scripts/train_aerial_transport_track.py --num-envs 32 --steps 20000
```
期望：都能跑完不报错（具体数值未复核）。

### 第 6 级：学习策略闭环（需要先拷贝检查点）
```bash
python scripts/serve_lerobot_policy.py outputs/train/act_so101_handle_grasp/checkpoints/last/pretrained_model
python scripts/collect_arm_grasp_demo.py --policy localhost:6000 --episodes 50 --seed 100
```
期望：约 40%（20/50）。同一种子多次运行会有几个回合的出入。

### 升级后最容易出问题的地方
代码里依赖了这些 Isaac Sim 4.1 / PhysX 的具体行为，新版本上先查它们：
- `omni.isaac.core`、`omni.isaac.sensor` 等旧命名空间。
- 浮动基座关节体的雅可比列布局（6 + 关节数），`EndEffector` 直接按此读取。
- `UsdPhysics.FixedJoint` 的 `excludeFromArticulation`，以及
  `RigidPrimView._physics_view.set_kinematic_targets`（私有接口）——采集脚本用它们固定并移动基座。
- USD 默认 prim 被合并进生成根节点（子节点直接出现在根下）。
- 多机器人同场景曾出现段错误、多环境构建时碰撞体烘焙死锁，当时都有规避写法。
- 驱动增益在 `_apply_asset_fixups` 里按 USD 的 `DriveAPI` 改写；SO-ARM 肩关节碰撞体与机身重叠的修复。

## 5. 对照 plan.md 的完成情况

| 阶段 | 条目 | 状态 |
|---|---|---|
| 0 | CTBR 接口 + 角速率内环 | 完成 |
| 0 | 电机时间常数与延迟 | 未复核，可能未做 |
| 0 | AerialTransport 抓取与双机共载闭链 | 部分：有脚本演示；真实接触抓取只在单臂采集场景验证过 |
| 0 | shoulder_pan 不响应 | 已解决（碰撞体重叠） |
| 0 | 数据格式（阶段标签、干净 / 带噪动作等） | 机械臂抓取数据已具备；无人机侧未统一 |
| 1 | 航点跟踪 RL，含挥臂 / 载荷 / 外力扰动，特权教师 | 未做。现有的是 AerialTransportTrack（跟踪演示）和未完成的 AerialGraspAssist |
| 2 | 可行基座集合 | 完成（下洗和相机视野两项过滤没做） |
| 2 | 规划代价加舒适度 | 未做（只有关节限位余量过滤） |
| 2 | 在 v0 策略控制的浮动基座上生成数据 | 未做：目前是运动学支架 + 人为漂移 |
| 2 | 视觉运动 IL 策略 | 已训练 ACT，成功率 40%，未达验收 |
| 3–6 | 联合迭代、多机协同、数据基准、多机型合并 | 未开始 |

### 阶段 2 的实验结论（未见过的基座位姿，各 50 回合）
| 策略 | 推理设置 | 成功 |
|---|---|---|
| ACT，关节状态 | 每次执行 25 步 | 20/50 |
| 同上 | 10 步 / 5 步 | 11/50、3/50 |
| 同上 | 时间集成 | 18/50 |
| ACT，加基座状态 6 维 | 25 步 | 22/50 |

脚本专家同场景 42/50。推理端调参和加基座状态都没有带来提升。

## 6. 下一步建议

1. **先做迁移验证**（第 4 节），确认新版 Isaac Sim 上第 0–4 级都过。
2. **补完阶段 2**：把抓取策略做到 80% 左右。两条已排除（推理设置、基座状态），剩下的方向是数据：
   采到约 1000 回合，并在采集时加大扰动让专家演示纠偏；先逐个看失败回合确认原因。
3. **并行启动阶段 1**：航点跟踪的无人机 RL（CTBR 输出）。它只需要臂的轨迹作为扰动，
   793 条可行基座下的规划轨迹已经有了，不用等抓取策略达标。同时把舒适度指标定下来
   （AerialGraspAssist 里已有雏形）。
4. **再回到阶段 2 的第 3 条**：用阶段 1 的策略替换运动学支架，在真正飞行的基座上重新生成数据并重训。
5. 之后按 plan.md 进入阶段 3（联合微调、教师蒸馏）。
