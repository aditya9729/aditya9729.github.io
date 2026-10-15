# Actual embodiment assets for the RL curriculum

Retrieved 2026-10-15 from the upstream repositories below. These are actual
published model geometries and inertial descriptions, not renamed generic
robots. Sources remain unchanged; generated local adaptations are named below.

| Local directory | Upstream source and pin | License | Native MuJoCo load target |
| --- | --- | --- | --- |
| `franka_panda` | [Google DeepMind Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35/franka_emika_panda) | Apache-2.0, retained in directory | `franka_panda/scene.xml` |
| `agilex_aloha` | [AgileX official Mobile ALOHA simulator, master](https://github.com/agilexrobotics/mobile_aloha_sim/tree/2843ff11d2695c1563a1c9847f632aa9734f5bbc/aloha_mujoco/aloha/meshes_mujoco) | MIT, upstream root LICENSE copied into directory | `agilex_aloha/aloha_v1.xml` |
| `galaxea_r1_pro` | [Galaxea official ManipSim](https://github.com/OpenGalaxea/GalaxeaManipSim/tree/abe7f5161eeaa150e6eaffdf443af5df7f23f356/galaxea_sim/assets/r1_pro) | Apache-2.0, upstream root LICENSE copied into directory | `galaxea_r1_pro/r1_pro_passive.xml`, generated here |

All three models were loaded with native `mujoco==3.3.7`, advanced for 100
simulation steps and produced finite state with no MuJoCo warning counts.
This checks loading and a brief numerical integration, not task success,
physical fidelity, hardware transfer, or robust long-horizon stability.

## Franka Panda

Menagerie derives this model from Franka's published URDF, and supplies position
actuators. `joint1` through `joint7` are arm hinges; `finger_joint1` and
`finger_joint2` are finger slides. There are 9 generalized positions and 8
controls, because the two fingers share a tendon actuator plus an equality
constraint. Arm controls are position targets in radians. **The native gripper
actuator is scaled to [0, 255], not radians or normalized [-1, 1].** The MJX
variant changes gripper actuation and solver settings; do not mix its control
conventions with `panda.xml`.

## AgileX ALOHA

This is AgileX's own legacy ALOHA MJCF on the repository's `master` branch. Its
current default `v2.0.0` branch is a ROS URDF description and does not contain
this MJCF. It is not DeepMind ALOHA 2 and not a dual-PiPER invention.

The arm joints are `fl_joint1` through `fl_joint6` and `fr_joint1` through
`fr_joint6`; finger slides are `fl_joint7`, `fl_joint8`, `fr_joint7`,
`fr_joint8`. There are 16 position controls: 12 arm + 4 fingers. Native arm
control targets are radians and finger targets are meters. `nq=29`, `nv=28`
also include an unnamed free-jointed soda can and a separate object with six
slide/hinge joints. Use named joint and actuator lookup, never assume the first
12 qpos entries are arms. The upstream example has wide arm joint ranges and
very large servo gains/force limits; these are simulator model settings, not
validated hardware limits. The chassis is static geometry with no mobile-base
joint or wheel controls, despite the repository's name.

The upstream Python scripts use obsolete `mujoco_py` and ROS for teleoperation.
The MJCF itself loads using modern native `mujoco`, without ROS or mujoco-py.

## Galaxea R1 Pro

`robot.urdf` is the actual official mesh, kinematic and inertial source.
`prepare_assets.py` copies it to `r1_pro_mujoco_import.urdf`, adds only MuJoCo
import compiler options (`meshdir=meshes`, `discardvisual=false`,
`fusestatic=false`) and saves `r1_pro_passive.xml` via `mj_saveLastXML`.
Original URDF and all mesh files remain unchanged. The generated model has
22 generalized positions and zero actuators:

- torso hinges: `torso_joint1` through `torso_joint4`;
- left/right arm hinges: `left_arm_joint1` through `left_arm_joint7`,
  `right_arm_joint1` through `right_arm_joint7`;
- four finger slides: `left_gripper_finger_joint1`,
  `left_gripper_finger_joint2`, `right_gripper_finger_joint1`,
  `right_gripper_finger_joint2`.

End-effector body frames are `left_gripper_link` and `right_gripper_link`.
`torso_link4` is the arm control reference used by Galaxea's IK settings.

**The base is fixed.** All three steering and all three wheel joints are
explicitly fixed in this published URDF. The official ManipSim implementation
uses SAPIEN and `loader.fix_root_link=True`. Merely importing this model does
not produce a validated wheeled whole-body controller. Any added position
servos, planar-base joints or velocity drives in a course wrapper are custom
educational choices and must be described as such.

The full official ManipSim stack lists Linux + CUDA GPU prerequisites and
`sapien==3.0.0b1`, `gymnasium==0.29.1`, MPLib, and external task assets. This
passive MuJoCo import can be inspected on CPU and is not the official task
environment. Galaxea also publishes an Isaac Sim 4.5 tutorial with chassis
movement in `source/wheel_move.py` and whole-body demo in
`source/whole_body_move.py`:
https://docs.galaxea-ai.com/Guide/R1Pro/isaacsim/r1pro_isaacsim_guide/

The ManipSim README says MIT, but the checked-out root LICENSE says Apache-2.0.
The retained actual license text is used here.

## Rebuild

```bash
python vendor/embodiments/prepare_assets.py
# Headless render on Linux, using installed EGL support:
MUJOCO_GL=egl python vendor/embodiments/render_assets.py
```

Rendered PNGs show geometry from MuJoCo forward kinematics at the source reset
configuration. They are not training rollouts or demonstrations.
