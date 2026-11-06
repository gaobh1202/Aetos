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
| 全驱动 6-DOF wrench 分配器 | ❌ 未实现,fahexa750 当欠驱动飞 |
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

这是固定倾角的全驱动六旋翼,6×6 wrench 矩阵 rank 6、条件数 4.2,**有**独立产生
侧向力的能力。但目前是用和另外五个一样的 4 自由度 mixer 飞的,侧向权限闲置。

它能接进现有 mixer 这件事本身需要说明:OmniDrones 的
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
机械臂:`1`-`9` 选关节,`[`/`]` 正反转,`G`/`H` 闭合/张开所有夹爪自由度,
`Z` 回零位。

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

---

## 11. 还没做的

- 全驱动 6-DOF wrench 分配器(fahexa750 的侧向权限现在闲置)。
- `uaquad600_gripper` / `fahexa750_gripper` 的 residual 没有扫过,只验证了够用。
- `fahexa750` authoring 的惯量 `(0.198, 0.198, 0.46)` 违反三角不等式。
- 外骨骼 / Apple Vision Pro 的真实硬件通路。SO-ARM leader 有代码,但没在真实
  硬件上测过。
- 5 个多机协作任务还是脚手架。
