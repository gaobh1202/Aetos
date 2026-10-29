# FAHexa 750 mm — MuJoCo 查看版

双击打开机器人.command。此目录包含全部XML、网格和启动脚本，可整体移动。
默认scene.xml为固定机体、零重力、运动学关节查看模式；Control滑块可调整原机械臂，R重置、G开合夹爪。
scene_free.xml为自由机体加重力版本，未提供飞控，直接运行会下落。
运行依赖见requirements.txt；当前电脑已有MuJoCo环境，其他macOS使用mjpython launch.py。

当前导出保留原USD装配，渲染材质为近似值，所有网格碰撞关闭。关节采用项目配置增益与力限幅，未迁移速度限幅；物理步长50微秒。
源几何、质量与关节报告见conversion_report.json，验证结果见validation.json；同目录CONVERSION_NOTES.md给出完整转换边界。
原Isaac Sim控制器、空气动力和任务环境未移植。

## 独立保存版本

750 mm 全驱动六旋翼，保留原上置机械臂和倾斜旋翼。
此文件夹可整体移动或复制，包含所有模型、网格、启动脚本、说明及许可证；不依赖制作目录或原Isaac Sim项目。运行需要MuJoCo/Python环境，当前电脑可直接双击“打开机器人.command”。
默认查看器固定机体，关节以运动学方式调节；自由机体版本未包含飞控，碰撞几何仍关闭。
package_check.json记录本独立目录加载检查，manifest.json记录文件SHA-256校验值。
