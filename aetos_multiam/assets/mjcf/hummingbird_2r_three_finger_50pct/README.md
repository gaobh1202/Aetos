# Hummingbird 340 mm + 50% 缩比两关节三指夹爪

本目录为当前确认尺寸的独立运行包。所有 XML、网格、控制参数和运行代码均在目录内，
可以整体移动或复制；不需要原始 amov_omni 项目、USD、机械臂目录或转换工具。
运行仍需 Python / MuJoCo 环境，不包含 Python 解释器及安装依赖。

## 文件

- `scene.xml`：MuJoCo 场景入口，包含地面、灯光和 robot.xml。
- `robot.xml`：完整无人机、50%缩比机械臂、夹爪及执行器模型。
- `assets/drone/`：由原 Hummingbird USD 导出的机体与四个旋翼网格。
- `assets/arm/`：机械臂 STL 网格；缩放记录在 XML 中，请不要再次缩放网格。
- `assets/gripper/`：夹爪 STL 网格；柔性手指的几何与材料直接定义在 XML 中。
- `launch.py`、`flight_control.py`、`flight_parameters.json`：交互悬停演示与控制配置。
- `打开仿真.command`：macOS 双击入口。
- `requirements.txt`：运行依赖，已验证 Python 3.10、MuJoCo 3.13.0。
- `assembly_preview.png`：当前尺寸的装配预览。
- `validation.json`：原目录中对同一模型进行的完整悬停/关节运动/开合测试结果。
- `package_check.json`：独立目录的资源与短时运行检查结果。
- `manifest.json`：随包文件的大小及 SHA-256 完整性清单。
- `LICENSE_OmniDrones`、`LICENSE_dynamixel_2r`：源资产许可。

## 启动

在当前电脑上双击 `打开仿真.command` 即可。启动器优先使用包内 `.venv`，
其次使用 PATH 中的 mjpython，最后尝试当前电脑已有的 dexsim 环境。

其他 macOS 电脑可在本目录用 Python 3.10 创建环境并运行：

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/mjpython launch.py
```

Linux/Windows 安装依赖后运行 `python launch.py`。已实际验证的是当前 macOS 环境。
若只需要模型，可以在其他程序中直接加载本目录的 `scene.xml`；悬停控制由 launch.py 提供。

Space 暂停/继续，R 重置，G 开合夹爪。右侧 Control 面板可以调节 R1/R2 目标角度。
旋翼输入由悬停控制器持续更新；夹爪输入由 G 键控制。

## 模型参数与边界

无人机保持 340 mm 轴距和原始尺寸；下挂机械臂、夹爪、转接件与挂载板线性缩比为0.5。
质量按等密度假设乘0.125，刚体惯量乘0.03125；这是缩比仿真假设，不是实物减重测量值。
整机质量约0.86414 kg，最大推重比约2.833。
R1/R2 保留 SO-ARM100 式位置伺服：kp=50、dampratio=1、armature=0.1、力矩限制±3.5 N·m。
无人机有6自由度，机械臂与机体刚性连接，夹爪保留三指柔性体和同步约束。

转接件和挂载板为仿真占位几何；没有完成制造孔位设计或全关节范围自碰撞验证。
悬停控制为基础验证演示，未包含气动下洗、电池、电机动态或原项目训练策略。
编译时原夹爪的3条 flex 提示仍会出现；已通过的运动测试中运行时数值警告计数为0。

源码来源：本地 amov_omni 的 Hummingbird USD/YAML、MuJoCo Menagerie 的 dynamixel_2r，
以及用户提供的 three_finger_gripper_final。源目录没有作为运行依赖保留。
