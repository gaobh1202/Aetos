# OmniDrones changes Aetos depends on

Aetos imports `omni_drones` from a fork: https://github.com/gaobh1202/amov_omni
at the commit in `BASE_COMMIT`, plus local changes that are not on that fork yet:

- `aetos.patch`: edits to tracked files (shared physics view for several robots in one
  scene, drones without rotor rigid bodies, the thrust + body-rate action transform, ...).
- `overlay/`: files that do not exist on the fork (controller gains for the Aetos drones,
  `amov_drone_arm_1.yaml`, the arm-trajectory / shelf-grasp envs and their configs, scripts).

Set up:

    git clone https://github.com/gaobh1202/amov_omni.git OmniDrones
    cd OmniDrones && git checkout "$(cat /path/to/Aetos_MultiAM/third_party/omnidrones/BASE_COMMIT)"
    bash /path/to/Aetos_MultiAM/third_party/omnidrones/apply.sh .
    pip install -e .
