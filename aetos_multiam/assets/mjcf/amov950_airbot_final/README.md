# AMOV 950mm quadrotor + Airbot Play arm (MuJoCo)

Source: `/home/benhua/raids-am-simulator/amov_drone_arm.xml` (Polyu-ISE RAIDS
Lab, copied with the user's go-ahead on 2026-10-29) -- that file is already
a standalone, single-document MJCF combining the drone body (`quadrotor_950`,
950mm-class, 6.5kg) with a 6-DOF Airbot Play arm + 2-finger tendon-coupled
gripper mounted underneath (`arm_mount_link`, euler-flipped 180deg).

`robot.xml` here is that file verbatim except mesh `file=` paths rewritten
from the source repo's absolute paths to this bundle's own
`assets/drone/`/`assets/arm/` (14 meshes: 1 STL for the drone body, 13 OBJ
for the arm+gripper -- same airbot_play geometry already used by this
project's own real USD asset at
`/home/benhua/DISCOVERSE/airbot_play_usd/my_airbot_play_v3_gripper/`, just
in MJCF form here instead). `scene.xml` wraps it with the same
floor/light/camera viewer convention as this directory's sibling bundles.

Joint names match the project's existing `airbot_play` manipulator spec
exactly (`joint1`..`joint6`, `endleft`/`endright`), and the drone's own
free joint is `main_joint` with actuator sites `thrust`/`rateX/Y/Z` --
same naming pattern AM-bench's own quad/hexa bundles in this directory use.

This is the 5th and last of this project's 5 named aerial-manipulator
presets (`amov950_airbot_bottom` in `robots/presets.py`) to get a real,
loadable MJCF -- the other 4 already had one from their own "viewer export"
bundles (`ambench_quad_600_top_arm_final`, `ambench_fa_hexa_750_final`,
`iris_650_so_arm100_final`, `hummingbird_2r_three_finger_50pct`); this one
didn't, since its real asset in this project was USD-only until now.
