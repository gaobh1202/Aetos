我们introduce Aetos，一个comprehensive simulation suite and benchmark for multi-aerial manipulator collaboration (MAMC) task.
Aetos包括了六种aerial manipulator，范围从欠驱动四旋翼到全驱动六旋翼，manipulator也覆盖2自由度到6自由度的常见构型，且包括two-fingered, three-fingered gripper, five-fingered dexterous hand.
为了降低硬件的门槛，我们相应地提供了一套现成地硬件配置（基于XX无人机+LeRobot SOArm）total expense小于XXX美元，可以参考-link-在1小时内完成组装。
Robots（Table，名称，图片，自由度，物理参数等）
quad950_airbot_gripper
quad950_airbot_dexterous
quad340_dynamixel_three_finger
quad600_arm_gripper
fahexa750_arm_gripper
quad650_soarm_gripper
我们首次提出了5种MAMC任务，并提供了超过5k条仿真合成数据，以支持前沿的whole-body control policy for aerial manipulation；任务涵盖：Two_AM_Scene (aerial_assembly, aerial_handover, aerial_transport)和Multiple_AM_Scene (aerial_cabinet_grasp, aerial_seq_assembly)
我们介绍了多种控制设备，包括keyboard/game pad/exoskeleton/apple vision pro等，从而轻松的控制各类aerial manipulator