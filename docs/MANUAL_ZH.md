# Aetos_MultiAM 中文手册

README 只讲"有什么、怎么跑第一条命令"。这份手册讲细节:现在验证到什么程度、
每个机型为什么这么配、踩过哪些坑、每个脚本的完整参数。

---

## 1. 环境

```bash
export ISAACSIM_PATH=/home/benhua/isaacsim
conda activate omnidrone

pip install -e /home/benhua/OmniDrones   # aetos_multiam 直接 import omni_drones
pip install -e .
```

两点容易踩:

- `conda activate omnidrone` 的激活脚本在没有 `ISAACSIM_PATH` 时会**静默空转**,
  新开的 shell 从来不会自带这个变量,所以顺序不能反。
- 不要用 Isaac Sim 自带的 `python.sh` —— 那个解释器里没装 `aetos_multiam` /
  `omni_drones`。

模型二进制(USD + 所有 MJCF bundle 的 mesh)不进 git,从网盘下载后解压到
`aetos_multiam/assets/meshes`,目录结构见 [`ASSETS.md`](ASSETS.md)。任务物体的
USD/mesh 同理,放在 `aetos_multiam/objects/assets/`。

---

## 2. 当前验证到什么程度

| 能力 | 状态 |
|---|---|
| 六个机型 spawn + 悬停 | ✅ 全部,悬停倾角 ≤ 1.1°,定点误差 3 cm 内 |
| 六个机型跑完 `show_am_action.py` 完整任务 | ✅ 单机 + 六机同场景,数值一致 |
| 键盘控制无人机 | ✅ 全部六个 |
| 键盘控制机械臂/夹爪 | ✅ 全部六个(两个已知打折,见 §4) |
| SO-ARM leader 实物遥操作 | ✅ 仅 `iris650_soarm100`(SO-ARM 系) |
| `ButtonGrasp` 任务端到端 | ✅ |
| `ShelfButtonGrasp` 任务 + 回放 | ✅ headless GPU PhysX 下跑完 605 帧,无 flip |
| `ShelfBucketGrasp` | ⚠️ 写法同上但没在 Isaac Sim 里启动过 |
| 5 个多机协作任务 | ⚠️ 只有注册和配置,是脚手架 |
| 全驱动 6-DOF wrench 分配器 | ✅ fahexa750 已用,水平飞行姿态 ≤ 0.23° |
| 外骨骼 / Apple Vision Pro 遥操作 | ❌ 未实现,只留了接入点 |

---

## 3. 六个机型,以及每个为什么这么配

命名规则 `<无人机尺寸+系列><机械臂><安装位置>`,定义在
`aetos_multiam/robots/presets.py`。这六个是**固定集合**,不支持运行时任意
组合 —— 组合机制本身存在(`utils/usd_compose.py`),但只对这六套源资产验证过。

### `amov950_airbot_bottom` — 四旋翼 950mm + airbot_play 6-DOF + 两指夹爪

整机(无人机+臂+夹爪)来自 OmniDrones 自己的 `amov_drone_arm_1.usd`,不是
`amov_950` + 单独 vendored 的 airbot 臂拼出来的。原因是那份 vendored 副本的
`arm_base` 质量是真实训练资产的 **4.7 倍**(0.777 kg vs 0.166 kg),而 PhysX
**不会**因为你改了 mass 值就重算缓存的惯量张量 —— 实测过:同一个训好的
hover policy,飞 vendored 资产会翻,飞真实资产不会。

`arm_dynamics_residual=0.001`。9.85 kg 机身挂 3 kg 的臂,反作用是真实物理,
对 residual 单调,必须压两个数量级:1.0 → 269° 偏航,0.1 → 98°,0.02 → 18.5°,
0.001 → 4.5°。

### `amov950_airbot_dexterous_bottom` — 同上,但换 Inspire 五指灵巧手

同一个 USD,spawn 时停用自带的两指夹爪(`link6` 及其 `gripper_left`/
`gripper_right`),把 Inspire Hand 挂到 `link5`
(`manipulators/amov_arm_bare.py` + `grippers/inspire_hand.py`)。

`arm_dynamics_residual=0.001`,这里**不是"更好"而是"必须"**:灵巧手的 12 个
手指驱动在 0.1 和 1.0 下会发散(关节位置先到 1e4 再到 1e12 rad,报
"Invalid PhysX transform",整个 articulation 变 NaN),0.001 下才稳。

`arm_write="teleport"`。0.001 只够让手"不动",一旦**命令**手指运动,无论
residual 多小,PD 驱动都会在第一个运动阶段发散 —— 资产把手指 authoring 成
`kp=0.05..0.07, kd=1e-4`,却和 `kp=1e7` 的臂关节在同一个 articulation 里,
跨 8 个数量级。按 residual 等比缩小增益试过,更糟(无人机飞到 19 m 高、
倾角 154°)。teleport 可以:完整任务跑下来机身倾角 0.46°。
**后续接 Apple Vision Pro 的 retargeter 也必须走 teleport 这条路。**

### `hummingbird340_2dof_gripper` — 四旋翼 340mm + 2-DOF 臂 + 三指夹爪

`arm_dynamics_residual=0.1`。这个几乎不需要隔离:完全耦合也只有 8.3° 横滚,
而且对 residual 线性,0.1 → 0.72°。

### `iris650_soarm100` — 四旋翼 650mm + SO-ARM100 5-DOF + 单爪

`arm_dynamics_residual=0.1` + `arm_locked_joints=("shoulder_pan",)`。

这个 residual **不单调**,0.1 是实测最优,0.02 反而差 8 倍。`shoulder_pan`
被锁是因为它的 drive 在场景里只要有另一个动态刚体在动就失效(见 §4),只剩
teleport 一条路,而 teleport 会把无人机偏航 67.8°。两条都在
[`ARM_DISTURBANCE.md`](ARM_DISTURBANCE.md) 里有完整测量。

### `uaquad600_gripper` — 四旋翼 600mm(AM-Bench)+ 4-DOF 臂 + 两指

`arm_dynamics_residual=0.1`,实测能悬停(1.96° 倾角)并飞完键盘序列,但这个
值**没有扫过**。

`joint_pose_overrides` 把臂从机身下方翻到上方 —— 见 §4。

### `fahexa750_gripper` — **全驱动六旋翼** 750mm(AM-Bench)+ 4-DOF 臂 + 两指

`arm_dynamics_residual=0.1`,同样实测可用但没扫过。`enable_self_collisions=True`
修穿模 —— 见 §4。

这是固定倾角的全驱动六旋翼,6×6 wrench 矩阵 rank 6、条件数 4.226,**有**独立
产生侧向力的能力。这个能力现在用上了 —— 见 §5bis。

它同时也能接进现有的 4 自由度 mixer(`--controller lee` 就是这么飞的),这件事
本身需要说明:OmniDrones 的
`compute_parameters(rotor_config, I)` 生成的是 4 行分配矩阵
`[sin(angle)*arm, -cos(angle)*arm, -direction*moment/force, 1]`。把 fahexa750
的真实几何代入反解等效 yaml 参数,残差 5.6e-17 —— 精确可表示,不是近似:

| 等效参数 | 值 |
|---|---|
| `arm_lengths` | 0.3637 |
| `rotor_angles` | [0.5233, 1.5707, 2.6183, -2.6183, -1.5707, -0.5233] |
| `directions` | [-1, 1, -1, 1, -1, 1] |
| `force_constants` | 2.8364e-05 |
| `moment_constants` | 6.7126e-06 |

两个约定上的转换:转子轴向量按 z 分量归一化;阻力矩单独用
`rotor_drag_ratio` 表达,不混进 `moment_constant`。其中 `ω_max` 是自己定的,
但它是**纯规范**:取 500 和取 838 得到的结果逐字节相同。

无转子关节的机型走 `robots/drones/rotorless_multirotor.py`,直接在
`base_link` 上施加合力/合力矩,不依赖 USD 里的转子 joint。

---

## 4. 已修复的八个缺陷

每一个都是真实缺陷,而且大多是**飞起来才发现的** —— 数值指标全过,只有在
视口里才看得出来。

| 现象 | 成因 | 修法 |
|---|---|---|
| `hummingbird340` 根本无法悬停(偏航 187.6°、飘出 21.1 m) | `ArticulationRootAPI` 加在 spawn root 上,PhysX 于是挑了机械臂的 `first_segment` 当浮动基座 | `robot_base._pin_articulation_root_to_drone()`,把 root 移到 `base_link` |
| `hummingbird340` 机械臂完全不动 | USD 把 `DriveAPI` 按**轴名**(`X`/`Y`)作为 multiple-apply 实例名,而驱动接口读的是 `angular`/`linear` | `utils/usd_fixups.normalize_joint_drives()`,把轴名实例复制一份到标准名 |
| 每个机型都比设定点下沉约 0.49 m | 控制器用的是 drone spec 里写死的质量,比组合后 articulation 的真实总质量少 16.5% | `initialize()` 从 articulation 读真实总质量并覆盖 |
| `uaquad600` 机械臂挂在机身**下面** | `quad_scorpion.usd` 把臂装在下方(实测每个臂体低于 `base_link` 0.05~0.29 m),而本项目同机型的 MJCF bundle 明确写了"翻转 180° 改为上置" | preset 的 `joint_pose_overrides`,用和 MJCF 完全相同的变换:pos (0.1, 0, 0.003)、quat (0, 1, 0, 0) |
| `fahexa750` 机械臂直接穿过机身 | `hexa_scorpion.usd` authoring 了 `enabledSelfCollisions=False`(另外五个都是 `True`),AM-Bench 自己的 ArticulationCfg 也要 `True` | preset 的 `enable_self_collisions=True` |
| Inspire 手指一命令运动就 non-finite | 资产把手指 authoring 成 `kp=0.05`,却和 `kp=1e7` 的臂关节同处一个 articulation | 该 preset 用 `arm_write="teleport"` |
| **两个机器人同场景直接 segfault**(崩在 `get_body_masses` 里,没有 Python traceback) | `initialize()` 里为修 rest pose 调的第二次 `sim.reset()` 是**全局**的,会销毁 physics simulation view,让其他机器人手里的 view handle 全部失效 | `initialize(reset_rest_pose=False)` + `reattach()`,由调用方统一 reset 一次 |
| 资产修复对"单独挂载的臂"静默失效 | `_apply_asset_fixups` 跑在 `mount_child` 之前,那时臂还不存在 | 移到 mount 循环之后的第二遍 |

### 多机同场景的正确初始化顺序

这个坑值得单独记,因为它**不抛异常,直接段错误**。一个机器人时完全看不出来,
两个就必然崩,三十行代码可复现:初始化 A、读 A 的质量(正常)、初始化 B、
再读 A 的质量 → 崩。

```python
sim.reset()
for rig in rigs:
    rig.am.initialize(reset_rest_pose=False)
sim.reset()                 # 整个场景只 reset 这一次
for rig in rigs:
    rig.am.reattach()
```

**同一个 bug 在 env 这条路径上也存在过**:`AerialManipulationEnv._setup_references`
原来就是一个朴素的 `for am: am.initialize()` 循环。单智能体任务看不出来;两个
智能体时,第二个的第一次写入就报 `Failed to get DOF positions from backend`,
紧接着段错误。现在那里用的是上面同一套顺序。

### 两个已知仍然打折的点

- `iris650_soarm100` 的 `shoulder_pan` 只能到达命令值的 2%。
- `hummingbird340_2dof_gripper` 的 `R1`/`R2` 只能到达 4%。

两个都是"drive 在有其他动态刚体时失效"的同一类现象,记录在
[`ARM_DISTURBANCE.md`](ARM_DISTURBANCE.md)。

---

## 5. 机械臂扰动隔离

动起来的机械臂会踹悬停中的无人机。把这件事做成可调的,是为了先用"简化的被控
对象"把飞控调通,再去学真实耦合动力学 —— 和人形机器人流水线里 retargeting 与
learned controller 的分工是同一个思路。

三个旋钮,都写在 preset 上,也都能从命令行覆盖:

| 旋钮 | 作用 |
|---|---|
| `arm_dynamics_residual` | 把臂的静态质量和惯量**合并到 `base_link`**(带平行轴修正),各连杆本身只保留这个比例的惯量。于是挥臂几乎不往机身传递反作用力旋量,而总质量、重心位置都还是对的 |
| `arm_locked_joints` | 直接锁死某个关节(目前只有 `iris650_soarm100` 的 `shoulder_pan`) |
| `arm_write` | `drive`(走 PD 目标)还是 `teleport`(直接写关节位置) |

`utils/arm_dynamics.py` 里写清楚了保留什么、不保留什么,以及为什么 residual 是
一个比例而不是一个固定的 epsilon。`--arm_dynamics coupled` 可以恢复完整多体物理。

实测效果:完整任务里,扫臂阶段对机身姿态的影响在**每个机型上都低于 0.7°**。

| preset | 扫臂最大倾角 | 全任务最大倾角 | 落地高度 | 写入方式 |
|---|---|---|---|---|
| `amov950_airbot_bottom` | 0.50° | 9.26° | 0.478 m | drive |
| `amov950_airbot_dexterous_bottom` | 0.46° | 9.58° | 0.478 m | **teleport** |
| `hummingbird340_2dof_gripper` | 0.74° | 9.70° | 0.379 m | drive |
| `iris650_soarm100` | 0.59° | 8.10° | 0.313 m | drive |
| `uaquad600_gripper` | 0.50° | 8.67° | 0.054 m | drive |
| `fahexa750_gripper` | 0.59° | 8.09° | 0.134 m | drive |

"全任务最大倾角"永远出现在最后的触地反弹,不是飞行中;飞行中最大的是"先左后右"
反向那一下的 8~9° 前倾,那是欠驱动机型本来就该有的姿态响应。

---

## 5bis. 全驱动分配:让六旋翼不用倾斜就能平移

### 为什么欠驱动机型必须倾斜

`LeePositionController` 只命令 4 个自由度:沿机体 z 的一个推力 + 三个力矩。
它的姿态目标因此**不是自由的** —— 是从它想要的力反推出来的
(`b3_des = -normalize(acc)`)。想往前加速,就只能把机体 z 轴往前倒。这对四旋翼
是物理必然,不是实现缺陷。

固定倾角的全驱动六旋翼不必如此。它的六个转子轴互不平行,张成的 wrench 空间
rank 为 6,力和力矩可以独立命令。

### 参考 AM-Bench 的做法

AM-Bench 在 `ambench/controllers/utils/control_allocation.py` 里实现:构造一个
6×N 矩阵,第 i 列是第 i 个转子单位推力贡献的 wrench,然后 `pinv` 一下打到期望
wrench 上。它的 `fully_actuated=True` 分支就是这个;`fully_actuated=False` 分支
把两行侧向力扔掉,退化成欠驱动的情形。配套的 `PID6DOFController` 位置环产生
**世界系下完整的 3 维力**(不投影到某根机体轴上),姿态环独立跟踪一个独立给定的
姿态目标,两者拼成 6 维 wrench。

它还有个 `_variable_tilt_allocation_matrix`,用 6×12 矩阵解
`F_i cos(α_i)` / `F_i sin(α_i)`,那是给**可变倾角**(转子带舵机)的平台用的。
我们的 fahexa750 是固定倾角,用不上,没有实现。

### 我们的实现

`aetos_multiam/controllers/drone/`:

- `allocation.py` —— 纯 torch,不 import Isaac Sim。`build_allocation_matrix()`
  构造 6×N 矩阵,`lateral_force_envelope()` 算侧向力包络。
- `fully_actuated_controller.py` —— `FullyActuatedPositionController`,
  继承 `LeePositionController` 复用它的增益读取/批处理,只换输出级。
  `make_position_controller(drone, g, mode)` 按 `drone.fully_actuated` 自动选。

和 AM-Bench 的一个关键区别:**矩阵的列是从本项目自己的 plant 约定推出来的**,
不是重新推导一遍。`RotorlessMultirotor.apply_action` 每个转子施加的是

```
力    = T_i * axis_i
力矩  = cross(r_i, T_i * axis_i) + (-dir_i * drag_ratio * T_i) * axis_i
```

所以列就是 `[axis_i ; cross(r_i, axis_i) - dir_i * drag_ratio * axis_i]`,一个
字都不多。这样分配器和仿真器**不可能**在符号上产生分歧。`dir_i` 用的是 yaml 的
`directions`(是自转方向的**取反**,这个符号藏在 `RotorGroup` 里,从资产重新推
很容易弄反 —— 第一版就弄反了,条件数从 4.226 变成 5.09)。

`T_i` 是转子推力的**竖直分量**而非推力大小,因为 `axis_i` 按 z 分量归一化而不是
单位化(见 `rotorless_multirotor.py`)。这正好是 yaml 的 `force_constants` 和
控制器 `max_thrusts` 携带的同一个变量,所以全链路不需要任何换算。

没有抄 AM-Bench 的积分项:这里复用的是 yaml 里已有的 PD 增益,加 I 项意味着四组
增益全部重调,不是三行代码的事。

### 侧向力必须钳位

分配只在每个转子都待在 `[0, max_thrust]` 区间内时才是精确的。超出之后求解器交给
PhysX 一个它会偷偷削掉的 wrench,而削掉的部分会耦合回姿态。

fahexa750 在悬停状态下、**最差方位角**能稳住的水平力是 **15.69 N = 0.241 g**
(最好方位角 20.03 N;最差在 120°)。而 `position_gain=4.0` 下 1 m 的位置误差
要求 26.5 N —— 大约是机体能给的两倍。所以 `max_lateral_force_n = 15.69` 写在
`ambench_750.py` 上,控制器据此钳位。结果是"平移慢一点",而不是"开始倾斜"。

这个数由 `allocation.py` 的 `lateral_force_envelope()` 算出,该模块的自检会重新
算一遍并断言(`python aetos_multiam/controllers/drone/allocation.py`,不需要
Isaac Sim)。

### 实测对比

同一个 preset、同一条任务、同一套增益,只改 `--controller`:

| 阶段 | `lee`(4 自由度) | `fa`(6 自由度) |
|---|---|---|
| 前进 1 m | 4.12° | **0.20°** |
| 左移 1 m | 4.28° | **0.22°** |
| 右移 1 m | 8.09° | **0.20°** |
| 偏航 180° | 2.95° | **0.23°** |
| 扫臂 | 0.59° | 0.53° |
| **全程最大倾角** | **8.09°** | **0.53°** |
| 平移中位置误差 | 0.303 m | 0.320 m |
| 稳态位置误差 | 0.058 / 0.073 m | **0.034 / 0.038 m** |
| 落地高度 | 0.134 m | 0.134 m |

全程最大倾角降了 15 倍,而且剩下的 0.53° 来自扫臂而不是飞行 —— 飞行段已经基本
看不到姿态变化了。平移时位置误差略大(侧向力被钳位,加速慢一点),稳态反而更准。

```bash
python examples/show_am_action.py --preset fahexa750_gripper                    # 自动用 fa
python examples/show_am_action.py --preset fahexa750_gripper --controller lee   # 退回欠驱动
```

`--controller {auto,lee,fa}` 在 `show_am_action.py` 和
`aerial_manipulator_control_demo.py` 上都有。`auto` 问 drone 的
`fully_actuated` 属性,但不轻信 —— 控制器构造时会真的算一次矩阵的秩,不够 6
就报错。

### 还没做的

姿态目标现在固定是"水平 + 命令偏航"。全驱动平台本来可以独立跟踪任意姿态
(比如保持机体倾斜去贴一个斜面),接口留着,但没有暴露出来。

---

## 6. 遥操作

### 设备矩阵

| preset | 无人机 | 机械臂(现在) | 机械臂(计划) |
|---|---|---|---|
| `amov950_airbot_bottom` | 键盘 ✅ | 键盘 ✅ | 外骨骼 |
| `amov950_airbot_dexterous_bottom` | 键盘 ✅ | 键盘 ✅ | Apple Vision Pro |
| `hummingbird340_2dof_gripper` | 键盘 ✅ | 键盘 ✅ | 键盘(就是最终形态) |
| `iris650_soarm100` | 键盘 ✅ | 键盘 ✅ / SO-ARM leader ✅ | 外骨骼 |
| `uaquad600_gripper` | 键盘 ✅ | 键盘 ✅ | 关节 / 末端 6D 位姿 + IK |
| `fahexa750_gripper` | 键盘 ✅ | 键盘 ✅ | 关节 / 末端 6D 位姿 + IK |

对比 AM-Bench:它的机械臂只有 **ee-centric** 一种控制方式
(`Se3Keyboard` → 7 维 → `AbsoluteEETargetIntegrator` → 8 维动作),没法单独控
某个关节;它代码里有个 `ABSOLUTE_BASE_JOINTS` 模式但没接上。我们这边是反过来
的:先有逐关节,末端 6D + IK 作为后续叠加。

### 按键

无人机:`WASD` 平移,`E`/`Q` 升降,`Left`/`Right` 偏航。
机械臂:`1`-`9` 选关节,`]`/`[` **或** `↑`/`↓` 正反转,`G`/`H` 闭合/张开所有
夹爪自由度,`Z` 回零位。

关节正反转给了两套键,因为 `[`/`]` 在非美式键盘布局上不在同一个物理键上
(德式要 AltGr+8/9),而 carb 读的是**物理键**不是字符 —— 方向键则是布局无关的。
设备在第一个机械臂按键被读到时会打印一行 `first key registered`,这样
"机械臂不动" 能立刻区分成 "键根本没进来" 和 "写入没落地" 两种情况。

机械臂这套绑定是**对 DOF 数量通用**的 —— 同一套键能开 2 自由度的臂,也能开
12 自由度的灵巧手(`devices/keyboard_arm_control.py`)。输入设备本身只做轮询
(每帧读键状态,不注册事件回调),和 `SOArmLeader` 是同一种写法。

### `--mode kinematic` 和 `--mode dynamic` 的区别

- `kinematic`:完全忽略转子动力学。键盘意图积分成一个目标位姿,每步直接传送到
  机体上。"飞一个空间中的点",机身永远不倾斜。
- `dynamic`:同一个目标位姿改由 OmniDrones 自己的 `LeePositionController` 在
  真实重力下、通过真实的逐转子推力(`drone.apply_action`)去追。实测(强制
  "前进+爬升",headless):机身向前俯冲(四元数 x 分量峰值约 -0.12),追上目标
  速度之后再回平 —— 这是几何控制律自己的副作用,不是手加的。

### `aerial_manipulator_control_demo.py` 完整参数

| 参数 | 取值 | 默认 | 说明 |
|---|---|---|---|
| `--mode` | `kinematic` / `dynamic` | `kinematic` | 见上 |
| `--preset` | preset 名 | — | 在空地面上加载这个 preset |
| `--task` | 场景名 | — | 用场景 yaml 里声明的机器人 |
| `--drone-model` | OmniDrones 机型名 | `Hummingbird` | 裸无人机,不带臂 |
| `--uav_control` | `none` / `keyboard` | `none` | |
| `--arm_control` | `none` / `keyboard` / `soarm_leader` | `none` | |
| `--arm-write` | `auto` / `drive` / `teleport` | `auto` | `auto` = 听 preset 的 |
| `--arm_dynamics` | `frozen` / `coupled` | `frozen` | |
| `--arm-residual` | float | preset 值 | 覆盖 |
| `--arm-lock-joints` | 逗号分隔 | preset 值 | 覆盖 |
| `--arm-joint-speed` | float | 0.8 | rad/s |
| `--arm-port` | 串口 | `/dev/ttyACM0` | leader 用 |
| `--arm-fake` | flag | — | 没硬件时用正弦扫描代替 leader |
| `--arm-recalibrate` | flag | — | 重新标定 leader |
| `--linear-speed` / `--z-speed` / `--yaw-speed` | float | 1.0 / 0.6 / 1.2 | |
| `--rate` | float | 50.0 | 控制频率 Hz |
| `--log` | 路径 | — | |
| `--headless` / `--max_steps` | | | |

三者选其一决定加载什么机器人:`--task <场景名>`(场景 yaml 自己声明,如
`ShelfBucketGrasp` → `iris650_soarm100`)、`--preset <名字>`(空地面)、
`--drone-model`(裸无人机)。

### 6bis. SO-ARM 夹爪的开闭范围(实测)

```bash
python debug/measure_gripper_range.py
```

`gripper` 关节行程 **−10° … +100°**(−0.1745 … +1.7453 rad),共 110°。换算成
两个指尖之间的距离:

| q | 开口 | q | 开口 |
|---|---|---|---|
| **−10°**(闭合) | **6.9 mm** | +45° | 83.2 mm |
| +2.2° | 23.9 mm | +63° | 105.7 mm |
| +14.4° | 41.5 mm | +82° | 125.5 mm |
| +26.7° | 58.7 mm | **+100°**(张开) | **142.0 mm** |

近似线性,**每度约 1.44 mm**。指长:动指 85.0 mm,定指 89.1 mm(自铰链量起)。
所以 Ø30 mm 的抓取柱只需要 +6.5°,余量极大。

量法上两个坑:

- **必须用 `set_joint_positions` 传送,不能用 PD 目标。** 实测命令 +31.5° 时 PD
  只到 +12°(开口 20.7 mm 而非 65 mm);而且一旦开重力,没有保持驱动的手臂会垮
  下来,量的就不是同一个姿态了。
- **只能取两个指头的 mesh**(`moving_jaw_so101_v1` 和
  `wrist_roll_follower_so101_v1`)。`gripper` 这个 link 上还挂着伺服舵机
  `sts3215_03a_v1`,它离铰链更远,会冒充"指尖"。

顺带一提:这把夹爪在资产上**其实是 SO101 不是 SO-ARM100**(Lightwheel 的
`so101_follower.usd` 替代件,见 `manipulators/so_arm100.py`),原尺寸、无缩放。

---

## 7. `show_am_action.py` —— 脚本化任务

在真实遥操作设备接上之前,用它当替身。它走的是**和键盘完全相同的**设定点积分
路径和机械臂写入路径,所以它跑通 ≈ 键盘能跑通。

```
静止 → 爬升到 1 m → 前进 1 m → 左移 1 m → 右移 1 m → 偏航 180°
     → 前进 1 m → 悬停,每个可驱动关节 +45° 回零 -45° 回零 → 降落
```

```bash
python examples/show_am_action.py --preset uaquad600_gripper
python examples/show_am_action.py --preset all                    # 六机同场景
python examples/show_am_action.py --preset all --headless --video out.mp4
```

几个实现上的决定:

- **平移是机体坐标系的**,所以偏航 180° 之后那次"前进"是沿原路返回。
- 45° 是**按每一侧**分别对该关节自身行程做钳位的 —— 两指夹爪的移动副关节能拿到
  它那几厘米的行程,而不是被对称钳位压成 0。钳位后的幅值会逐关节打印。
- 还会打印每个关节扫描中**实际到达的峰值行程**。这是唯一能区分"动了又回来"和
  "压根没动"的办法 —— hummingbird 那条没驱动的臂,就是这样连续几轮飞行测试都
  没被发现的。
- `--preset all` 把六个机型在同一地面上间隔 4 m 摆开同时飞,每个带自己的控制器、
  设定点、residual 和写入模式。测出来的数值和单机跑完全一致,说明互不干扰。
- 落地检测要同时满足三个条件(处于 landing 阶段、设定点低于机体 0.02 m、
  连续 0.3 s 在 5 mm 内磨蹭)。早期版本只看 `vz<0` 会漏掉落地后的静止段;
  只看"设定点低于机体"会在爬升过冲时误触发,把设定点冻在 1.02 m。

参数:`--arm-residual` / `--arm-write` / `--arm-sweep-deg`(默认 45)/
`--spawn-z`(0.5)/ `--cruise-z`(1.0)/ `--leg`(1.0)/
`--linear-speed`(0.5)/ `--z-speed`(0.4)/ `--yaw-speed`(0.8)/
`--arm-speed`(0.6)/ `--rate`(50)/ `--video` / `--capture-every`(4)/
`--video-fps`(25)/ `--headless`。

README 里那六个动图就是 `--video` 出来的 mp4 转的 3 倍速 GIF。mp4 本身
(`resources/*.mp4`,每个约 11 MB)不进 git。

---

## 8. 任务

### `ButtonGrasp`

950mm AmovDrone + airbot_play 飞到墙上的按钮前按下去。端到端验证过。

```bash
python scripts/run_button_grasp.py headless=true
```

### `ShelfButtonGrasp` / `ShelfBucketGrasp`

四层镀铬货架,上面放齿轮、按钮、电钻、周转箱,从 raids-am-simulator 的
`shelf_grasp.usd` 迁过来(`scenes/shelf_arena.py` + `objects/{gearwheel,
shelf_button,power_drill,tote_box}.py`)。

```bash
python scripts/run_shelf_button_grasp.py task=ShelfButtonGrasp headless=true

# 换机器人,不改代码
python scripts/run_shelf_button_grasp.py task=ShelfButtonGrasp \
    aerial_manipulators.0.preset=amov950_airbot_dexterous_bottom headless=true
```

### `AerialAssembly` —— 两台 AM 各取一个齿轮插进同一块底板

场景是 `WarehouseArena` + 混凝土地面(§8bis)、**不要龙门架**
(`WarehouseArena(gantry=False)`):

* 两个黄框正中间立一个**方柱**(截面和底板一样大,高 0.65 m),顶上装 IsaacLab 的
  `factory_gear_base` 齿轮底板,三根轴朝上;
* 圆柱旁边一张**更低的桌子**(台面 0.60 m 高、1.20 × 0.50 m),两端各放一个齿轮;
* 两台 `iris650_soarm100`,各在一个黄框正中(`(±4.5, 0, 1.2)`,从
  `arena.spawn_zone_centers` 读,不硬编码)。

每个智能体飞到桌子自己那一端取走齿轮,运到圆柱上方,沿轴放下去。用的是
**medium + small** 两个齿轮 —— 它们落在相邻的两根轴上,座好之后正好互相啮合,
所以"都插进去"和"啮合上"是同一个终态,不是两个条件。

### 尺寸:`gear_scale = 4` + 固定尺寸的抓取柱

`task.gear_scale` 默认是 **4** 不是 1。IsaacLab 原尺寸的齿轮只有 Ø42 / Ø22 mm,
放在 18 m 的厂房里、650 mm 的无人机旁边等于看不见。4 倍之后:

| | 1× | 4× |
|---|---|---|
| 底板 | 150 × 75 mm | **600 × 300 mm** |
| 轴 | Ø9.56 × 20 mm | **Ø38 × 80 mm** |
| medium 齿轮 | Ø42 × 25 mm | **Ø168 × 100 mm** |
| small 齿轮 | Ø22 × 25 mm | **Ø88 × 100 mm** |
| 两轴间距 | 30.5 mm | **122 mm** |

放大之后齿轮本体早就超出夹爪行程了,所以每个齿轮的轴心上**立了一根固定尺寸的
抓取柱**:Ø30 mm × 80 mm,`FactoryGear.grab_post_diameter_m` /
`grab_post_length_m`。夹爪闭合的是这根柱子,不是齿轮本体 —— 于是"物体放多大"和
"夹爪够不够得着"彻底解耦。Ø30 mm 对应 SO-ARM 夹爪约 **+6.5°**,而它有 110° 行程
(见 §6bis);80 mm 的柱长也刚好在 85 mm 的指长之内。

实现上两个细节:柱子是作为**齿轮刚体的子 prim** 建的(加 `CollisionAPI`,不另建
刚体、不加关节),所以它天然跟着齿轮走;而它的半径和高度要**除以 spawn 的
scale**,因为 scale 挂在上一层包装 prim 上,否则柱子会跟着齿轮一起放大 ——
那就白做了。齿轮的质量是资产里写死的 50 g,加一根柱子不会改变它。

方柱和桌子的高度也跟着调了(方柱 0.60 × 0.30 × 0.65 m —— 截面跟着
`gear_scale` 走,始终和底板一样大;桌面 0.45 m),让抓取柱落在悬停在 1.2 m 的
无人机手臂够得着的高度。

每个智能体的目标点改成了它自己那根轴正上方,不再是场地中心。

```bash
AETOS_SMOKE_TEST_STEPS=30 python scripts/run_collab_smoke.py task=AerialAssembly headless=true
python scripts/check_gear_assembly.py headless=true     # 验证真能插进去且啮合
```

`check_gear_assembly.py` 不用机器人:把两个齿轮依次传送到各自轴的上方松手
(高度按 `gear_scale` 缩放),等它们落位,然后把其中一个**一次性**转起来、看另一个
转不转。4× 下实测:

```
gear_0_medium: dz=+0.0007 m  dxy=0.0059 m  tilt=4.18 deg
gear_1_small : dz=+0.0022 m  dxy=0.0026 m  tilt=0.91 deg
driven medium 转到 +4.00 rad/s -> follower small 峰值 0.404 rad/s(反向)
```

啮合测试在 4× 下改成了**给一次初速度再让它自己滑行**:原来那种"每步强制设定
速度"在大尺寸下完全失效(接触求解器把指令吃掉,两个齿轮都停在 0.01 rad/s),
看起来像没啮合,其实只是驱动方式不对。

三个踩过的坑,都写在 `objects/factory_gear.py` 里:

1. **每个齿轮资产都带 `ArticulationRootAPI`**(IsaacLab 自己用 Articulation 包装
   器驱动它们),PhysX 于是把它当成单连杆关节体而不是自由刚体,`RigidPrimView`
   一无所获 —— 报 "did not match any rigid bodies",没有任何 cooking 报错能指向
   它。底板没有这个 API,所以底板一直是好的。`strip_articulation_root()` 解决。
2. **几何偏置烘进了网格顶点**:每个齿轮的网格中心都在它该去的那根轴正上方
   (large −30.25 mm / medium +20.25 mm / small +50.75 mm,底面在 z=+5 mm),
   所有 xform 都是单位阵。所以把齿轮生成在底板**同一个坐标**上就自动套在轴上;
   放到别处必须减掉这个偏置(`resting_translation()` 干这件事)。
3. **松手高度必须小于 20 mm**:孔从物体原点往上 5 mm 开始、轴高 25 mm,放得更高
   孔就完全脱离轴,齿轮只会倒在板上。早期用 20 mm 时两个都倒了(倾角 57° / 102°)。

另外两个齿轮**必须一个一个放**:同时松手会齿对齿卡住。放的时候给一点绕轴自转
让齿自己找相位 —— 这个自转速度要**按 `gear_scale` 反比缩放**:同样是 2 rad/s,
对 42 mm 的齿轮是轻轻一拨,对 168 mm 的直接把它甩下轴(实测偏心 20.6 mm、倾角
14.6°)。这和真实装配是一回事。

`env_spacing` 从 8 提到 24:每个 env 现在是一整栋 18 × 20 m 的厂房,8 m 会重叠。
`num_envs` 降到 1,因为这个任务还没有真实的抓取/插入奖励(奖励和成功判定仍是
`CollaborativeAerialEnv` 的占位实现),没人在训它。

### `AerialHandover` —— 一台取锤子、交给另一台、丢进箱子

场景同样是 `WarehouseArena` + 混凝土地面、不要龙门架:

* 两个黄框之间的空档(|x| 2.6~6.4 之外的 5.2 m)里摆**两张桌子**,各在一台机器人
  面前:`x = -2.0` 的桌上放一把锤子,`x = +2.0` 的桌上放一个周转箱;
* `iris650_soarm100` 在 −x 框,`uaquad600_gripper` 在 +x 框。

设想的流程:SO-ARM 那台把锤子从桌上拿起来,两台在场地正中会合完成递交,
再由夹爪那台丢进箱子。这条流程需要的三个航路点
(`hammer_pickup_position` / `handover_position` / `box_drop_position`)是
env 上的 property,不是埋在奖励里的数字 —— 以后无论谁来驱动机器人都从同一个
地方读。

**故意做成异构的**:5 自由度臂 + 单爪 递交给 4 自由度臂 + 平行夹爪,这是同构
编队永远测不到的情况。

```bash
AETOS_SMOKE_TEST_STEPS=30 python scripts/run_collab_smoke.py task=AerialHandover headless=true
```

物体:锤子是 NVIDIA SimReady 的 `handTool_hammer_crossPeen_b01`
(268 × 104 × 36 mm,**627 g**),箱子是 shelf 场景里就有的 `ToteBox`
(600 × 400 × 170 mm,开口朝上)。627 g 挂在 SO-ARM100 末端,真实伺服是扛不住
的,仿真里靠 `arm_dynamics_residual` 才拿得动 —— 这一点要记着。

实测摆放稳定:锤子落位后只动了 2.0 mm,箱子 0.0 mm。

抓取、递交本身、以及任何知道这些阶段的奖励**都还没实现**,奖励和成功判定仍是
`CollaborativeAerialEnv` 的占位实现(两台都靠近共同目标点,目标点设成递交点)。

#### 两个新修的缺陷

1. **SimReady 道具是 `instanceable` 的,PhysX 不给它建刚体。** 锤子的
   `hammer_01_obj_00` 上明明有 `PhysicsRigidBodyAPI`,碰撞网格却在 instance
   prototype 里,普通的 stage traverse 根本看不到它(要用
   `Usd.TraverseInstanceProxies`),而 PhysX 对 instanceable 的 prim 不会建
   per-instance 的 actor —— 症状又是那句 "did not match any rigid bodies"。
   `utils/usd_fixups.make_non_instanceable()` 把整棵子树的 instanceable 清掉。
2. **`reset_arm_joints` 本来挂在 drone 上,对非 Amov 机型直接崩。**
   `CollaborativeAerialEnv._reset_idx` 调的是 `am.drone.reset_arm_joints()`,
   而只有 `AmovDrone` 有这个方法(它的臂烘在自己的 USD 里);其他 preset 的臂是
   运行时挂上去的,drone 类压根不知道有臂,于是报
   `'_AmbenchQuadScorpion' object has no attribute 'reset_arm_joints'`。
   现在方法挪到了 `AerialManipulatorBase` 上(回零位,并按各关节自身限位钳位 ——
   airbot 的 `joint3` 范围是 [-0.094, 3.161],不钳位就会初始化到限位之外)。
   顺带把一直跑不起来的 `AerialCabinetGrasp` 也修好了。

### `AerialTransport` —— 两台一起抬一个箱子

场景同样是 `WarehouseArena` + 混凝土地面、不要龙门架,两张桌子都在 x = 0、
沿 Y 排开:

* **取件桌**(y = 0)上放一个**双提手箱**,长轴沿世界 X,两个提手因此落在 ∓x,
  一台机器人一个;
* **放置桌**(y = 2.5)是空的,台面上用**绿线**画了一个矩形,标出箱子该落在哪;
* 两台 `iris650_soarm100`,各在一个黄框里。

**故意用同构的两台**(和 `AerialHandover` 相反):两点起吊只有在两端行为一致时
才说得通。

```bash
AETOS_SMOKE_TEST_STEPS=30 python scripts/run_collab_smoke.py task=AerialTransport headless=true
```

航路点同样是 property:`pick_table_center` / `drop_table_center` /
`handle_position(index, on_drop_table=False)`。**索引 0 是 −x 提手,也正是在
−x 黄框里出生的那台机器人** —— 两个顺序是故意对齐的。实测:

```
handles (pick) [(-0.42, 0.0, 0.79), (0.42, 0.0, 0.79)]
handles (drop) [(-0.42, 2.5, 0.79), (0.42, 2.5, 0.79)]
carry_box 落位后移动 0.0 mm
```

#### 提手箱是拼出来的,不是找来的

`objects/carry_box.py`。资产库里没有"一个刚体 + 两个已知间距的抓取特征"这种
东西,所以直接用图元拼:根 Xform 带 `RigidBodyAPI`/`MassAPI`,箱体和两个提手
作为子 prim 各带 `CollisionAPI` —— PhysX 看到的是**一个 actor、多个 shape**,
和 IsaacLab 齿轮资产的结构、以及 `FactoryGear` 那根抓取柱用的是同一招。

箱体 0.90 × 0.40 × 0.22 m、1.0 kg(两条臂分摊,各 0.5 kg)。每个提手是个"U":
两根短柱从箱顶靠近短边处立起,上面架一根横杆,横杆沿 Y。夹爪闭合的是**横杆**,
Ø30 mm —— 和齿轮抓取柱同一个尺寸,对应 SO-ARM 夹爪约 +6.5°(见 §6bis)。
横杆离箱顶 120 mm,够夹爪从上方套进去而不碰到箱体。

提手做成**朝上的 U** 而不是嵌在短边立面上,是因为 aerial manipulator 的臂是向下
挂的,只能从上方抓 —— 立面上的提手它够不着。

绿色矩形是纯视觉的(箱子 footprint 加 80 mm 余量),和黄框一样不带碰撞:放箱子
的时候台面上多出 2 mm 的台阶迟早会出接触问题。

抓取、双点起吊的闭链动力学、以及对应的奖励**都还没实现** —— 真正刚性的两点
搬运需要一个闭合运动链(两个固定关节连到同一个刚体),那是另一件工程。奖励仍是
占位实现,但目标点设在了**放置桌上各自提手的正上方**,至少指向是对的。

### `examples/run_task_ShelfButtonGrasp.py` —— 带真实控制源

货架场景真正当成**任务**跑:场景 + 一个东西飞无人机 + 一个东西动机械臂,
两边的控制源各自独立显式指定。

- `--drone-control {policy,device}`(默认 `policy`):`policy` 加载 OmniDrones
  那边 `task=HoverAmovDynaArmReplay` 训出来的 hover checkpoint;`device` 是
  预留接口(抛 `NotImplementedError`)。
- `--arm-control {replay,policy,device}`(默认 `replay`):`replay` 回放迁移过来
  的 raids-am-simulator 录制片段(臂关节 + 夹爪 + 4 个物体位姿,放在
  `aetos_multiam/data/shelf_button_grasp/episodes/`);另外两个是预留接口。

**让无人机不翻,需要六个修复**,都值得记住:

1. 运动学属性必须在 GPU pipeline 下的**第一次** `sim.reset()` 之前 authoring。
2. 道具内部的每一个 joint 都要先停用,再把它变成 kinematic。
3. 机械臂的关节**位置和速度**(不只是 PD 目标)要在第一个物理步之前传送到回放
   的起始帧。
4. 录制轨迹是 30 fps,要重映射到仿真自己的控制频率(约 62 Hz),并且在 hover
   policy 稳定之前把臂按在第 0 帧。
5. 臂的 PD 增益要和 checkpoint 训练时一致(stiffness=500 / damping=50)。
6. policy 的原始动作要过一遍 `hover.py` 的 `_pre_sim_step` 同款仿射重映射
   (`action*scale+offset`)—— 跳过它就是一个静默的、系统性的推力偏置。

已经和 OmniDrones 自己的 `scripts/run_shelf_grasp.py` 做过对照验证(同
checkpoint、同 `ep000.npz`):两边都能跑完 605 帧,零 PhysX/Python 报错,
不翻,都能从中段一次大扰动(pos_err 冲到 1.2~1.8、`up_z` 掉到 0.93~0.96)
里恢复。还剩一个纯外观的缺口:按钮的两个装饰性子按钮会冻在 spawn 的位置
(它们的 joint 没了,也没有别的东西驱动)。

#### 为什么这个脚本不用 preset 系统

aetos_multiam 整个存在的意义就是可换的 drone × arm × gripper 运行时组合,
省掉每种组合手工烘一个 USD。但那恰恰是**这个脚本**的反面需求:它要精确复现
某个已训练 checkpoint 的物理配置,而预烘的合并 USD 免费给你这个保证(同一个
文件、逐字节一致),运行时拼装单独 vendored 的部件做不到(见 §3 里的质量/
惯量不匹配)。

想换组合、或者不需要对齐已有 checkpoint → 用 preset 系统;需要对齐 → 用直接
加载 USD 的写法。

---

## 8bis. 场景

`aetos_multiam/scenes/` 下每个 `Arena` 只负责填充**一个 template env** 的内容,
env 原点的网格复制由 `IsaacEnv` 的 `GridCloner` 负责(从 OmniDrones 继承)。
看某个场景(不放机器人):

```bash
python examples/show_scene.py --scene default
python examples/show_scene.py --scene warehouse
python examples/show_scene.py --scene warehouse --floor concrete --headless
```

### `Arena` —— 默认场景

一块地面,别的什么都没有,其他场景都从它继承。地面是 Isaac 自己的
`Environments/Grid/default_environment.usd`,染成白色 —— 这正是 AM-Bench 的做法
(`spawn_ground_plane(cfg=GroundPlaneCfg(color=(1.0, 1.0, 1.0)))`),同一个 USD、
同一种染色,所以渲染出来就是他们那个蓝底白格的地面。OmniDrones 对同一个调用的
默认值是近黑的 `(0.065, 0.0725, 0.080)`,之前那六个任务视频就是在那个地面上录的。

### `WarehouseArena` —— 仓库

18 × 20 × 5.4 m 的封闭厂房,四面墙 + 墙裙 + 蓝色壁柱;地面上两个 3.8 × 4.0 m 的
黄色框,中心在 (±4.5, 0);两框**正中间**一个四柱龙门架,跨度 8 m(立柱在
y = ±4)、进深 1.6 m(立柱在 x = ±0.8)、大梁在 4.43 m,梁上两个小车位。

布局参考的是 `industrial_warehouse_mujoco/build_scene.py`,**有一处故意改了**:
那边龙门架在两个黄框的**后面**、横跨它们;这里改成在两框**中间**,大梁沿 Y 走向
——因为两个框是两台 aerial manipulator 的起飞区,龙门架是它们汇合的地方,所以
四个立柱必须落在两个框之外、不挡航道。

几个实现上的决定:

- **黄线是纯视觉的**(`VisualCuboid`,无碰撞),和 MJCF 里的
  `contype=0 conaffinity=0` 一致。2 mm 的台阶卡在起落架下面是迟早要出的接触 bug。
- **`floor_color` 默认 `None`**,也就是直接用 AM-Bench 那块地面。给它一个 RGB
  (混凝土是 `(0.70, 0.70, 0.68)`)会**替换**掉地面而不是盖在上面 —— 盖在上面会
  整片 z-fighting,而把 grid plane 的可见几何隐藏起来并不可靠(网格不是由 Mesh
  prim 画的)。替换之后这块板本身是碰撞体,所以地面是有限大的 —— 封闭厂房里无所谓。
- **USD prim 名不能带 `-` 和 `+`**,所以正负号在名字里写成 `p`/`n`。
- **Kit 的视口会自己画一层参考网格**(还有红绿世界轴),叠在地面上看起来像地面
  自带的纹理。`show_scene.py` 把 `/app/viewport/grid/enabled` 关掉了。
- **Kit 的透视相机住在 session layer**,用 `prim.GetAttribute().Set()` 改
  `focalLength` 会"成功"但没效果,得走 `ChangeProperty` 命令。默认 18.1 mm 的镜头
  在屋子里根本装不下整个场景,warehouse 预设用的是 10 mm。

背景杂物(货架、纸箱、后墙卷帘门)没做 —— 那是布景,任务不碰它。MJCF 那份烘的
混凝土贴图也没有对应物,除非专门写一个材质。

龙门架下面的挂点坐标是 `WarehouseArena.hang_anchor_positions`
(`(0, ±1.6, 4.25)`),起飞区中心是 `spawn_zone_centers` —— 任务从这两个属性读,
不要再硬编码一遍。

---

## 9. 资产导出

`assets/meshes/usd/combined/` 下的每个文件(除了 `uaquad600_gripper.usd`,
它需要专门的重挂载手术,有自己的脚本)都由 `scripts/export_robot_usd.py`
按一份 YAML 配置生成:

```bash
python scripts/export_robot_usd.py --config scripts/export_configs/iris650_soarm100.yaml
```

一个输出文件对应一份配置,都在 `scripts/export_configs/` 下。配置格式见
`export_robot_usd.py` 自己的 module docstring,什么时候该重跑哪一份见
[`ASSETS.md`](ASSETS.md)。

---

## 9bis. `scripts/` 和 `debug/` 的分界

- **`scripts/`** —— 和具体任务或算法强相关的:任务入口
  (`run_button_grasp.py`、`run_collab_smoke.py`、`run_shelf_*.py`)、资产导出
  工具(`export_robot_usd.py`),以及**绑定到某个任务**的验证脚本 ——
  `check_gear_assembly.py` 会直接构建 `AerialAssembly` 环境、断言那个任务自己的
  要求(两个齿轮都座住且啮合),所以它和任务入口放在一起。
- **`debug/`** —— 只回答"这个资产/仿真器本身是不是我们以为的那样",不碰任务也不碰
  算法,benchmark 里没有任何代码 import 它们:
  `measure_gripper_range.py`(夹爪开口 vs 关节角)、
  `verify_iris650_flight.py`(iris650 改完资产后还推不推得动、悬不悬得住)、
  `record_iris650_trajectory.py`(把一段脚本飞行录成 PNG 序列)。

---

## 10. 其他容易踩的坑

- **Isaac Sim 会把 USD 的 default prim 合并进 spawn root**,所以子 prim 直接
  出现在 spawn root 下面,不会多一层。写路径表达式时注意。
- **横滚角报 ≈ 360° 是角度缠绕**(重置时参考角在 ±π 附近),不是翻机。
- **灾难性的探测用例会污染同进程后续的 `env.reset()`**。扫 residual 这类参数
  时,一个配置一个进程,或者只信第一行结果。
- **teleport 写入的关节不会自己回零**:没有驱动把它顶回去对抗邻居。Inspire 手
  扫完 +45° 之后 thumb_3 停在 +18.8°、index_2 停在 +21.9°。
- **系统的 ffmpeg 可能没有 libx264**,录视频时脚本会退回 `imageio_ffmpeg` 自带
  的二进制,再不行退 mpeg4。
- **在渲染脚本里用 `set -u` 会打断 conda 的激活钩子**(`PYTHONPATH: unbound
  variable`),整批任务静默产不出东西。
- **`instanceable` 的 prim 不会成为 PhysX 刚体**。NVIDIA SimReady 道具默认就是
  这样 authoring 的:碰撞网格在 instance prototype 里,普通 traverse 看不见,PhysX 也
  不建 actor。用 `make_non_instanceable()`。
- **"did not match any rigid bodies" 有至少三种成因**:prim 上挂了
  `ArticulationRootAPI`(IsaacLab 的齿轮)、prim 是 `instanceable`(SimReady 道具)、
  或者 prim 路径本身写错了。三种的报错一模一样,而且都没有别的日志能区分。

---

## 11. 还没做的

- 全驱动平台的任意姿态跟踪(现在姿态目标写死为"水平 + 命令偏航",见 §5bis)。
- 可变倾角(转子带舵机)平台的 6×12 分配,AM-Bench 有,我们没有对应机型。
- `uaquad600_gripper` / `fahexa750_gripper` 的 residual 没有扫过,只验证了够用。
- `fahexa750` authoring 的惯量 `(0.198, 0.198, 0.46)` 违反三角不等式。
- 外骨骼 / Apple Vision Pro 的真实硬件通路。SO-ARM leader 有代码,但没在真实
  硬件上测过。
- 5 个多机协作任务还是脚手架。
