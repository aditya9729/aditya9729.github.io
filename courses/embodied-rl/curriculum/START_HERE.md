# Reinforcement learning → deep RL → robot learning

This is an 18-chapter curriculum, not a simulator installation course.
Each chapter develops visual intuition, derives the mathematics, works numbers,
shows editable code, and ends with an experiment and an explanation check.
The notes contain about 15,500 words. Plan roughly 60–120 minutes per chapter
including the lab; take longer for derivations or extended experiments.

## Start with probability, not PPO tuning

1. Open `notebooks/00_probability.ipynb` in JupyterLab.
2. Read the narrative and predict the gradient before running the worked case.
3. Draw the result yourself, then compare it with the notebook plot.
4. Change one quantity, rerun, and record prediction → change → result → explanation.
5. Continue through the chapters in order. The tiny symbolic examples are exact
   decision models for algebra; they are never labeled physical robot benchmarks.

The website is the illustrated reader; Jupyter is the editable laboratory.
The complete download includes an offline reader in `website/index.html`.
ZIP download links within that offline copy require the hosted site; local
notebooks and notes are already present in the extracted folder.

## Curriculum

| Unit | Chapters | What you should be able to do |
|---|---|---|
| I: Foundations | 00–03 | Interpret distributions, specify an MDP, derive Bellman recursion, distinguish MC/TD targets and episode boundaries |
| II: Deep value learning | 04 | Implement a neural Q update, replay and detached targets; diagnose selection bias |
| III: Policy optimization | 05–07 | Derive REINFORCE, actor–critic/GAE and all four PPO clipping cases |
| IV: Off-policy control | 08–10 | Follow deterministic actor gradients, implement TD3 and SAC updates, check action densities and entropy |
| V: Robot learning | 11–14 | Distinguish BC/offline/online RL, compose residuals, run real Franka/AgileX/R1 Pro model tasks |
| VI: Research literacy | 15–17 | Assess GRPO/flow policies, curricula/rewards/retention, systems claims and a controlled capstone |

## Mac / CPU

Use a supported Python/PyTorch combination for your Mac architecture. Python 3.11
is a conservative starting point. Intel and Apple Silicon wheel availability
differs; the broad Torch range lets pip select a compatible available version.
The code does not require CUDA or Metal/MPS.

```bash
python3 scripts/install.py --python python3.11 --profile sb3
.venv/bin/python -m jupyter lab notebooks/00_probability.ipynb
```

Run these from the extracted `Embodied_RL_Curriculum` folder. Desktop rendering
uses the GL backend supported by your OS. Do not set `MUJOCO_GL=egl` on a Mac
just because it was used on headless Linux. No Mac-native execution was available
for this build, so the instructions are a provided path, not a tested Mac result.

## Google Colab / optional GPU

1. Use a chapter's Open in Colab button. It names the actual public GitHub notebook.
2. Colab opens that chapter; sign in and confirm third-party execution if asked.
3. Run the first cell. It restores embedded editable Python modules when needed.
   No curriculum ZIP upload or private-site authentication is required.
4. The setup keeps Colab's existing CUDA-capable Torch. It installs course
   essentials and uses CPU by default. Set `USE_GPU=True` only if CUDA is allocated.
5. Download your edited notebooks, logs, and models before your session ends.

The preview website is owner-private; the notebooks are separately available in
the public GitHub review branch using the standard Colab GitHub URL format.
GPU allocation and runtime limits are controlled by Colab. Small state networks
with one CPU simulator can be slower on a GPU because of transfer/synchronization.
The CUDA code path is included but was not executed on a Colab GPU here.

## Framework workbenches and publication

Read `INSTALL.md` for executable core/SB3 installers, a dependency doctor,
the same-task SB3 baseline runner, and separate pinned LeRobot/RLinf source
installers. Those advanced installations are optional and were not executed here.
Read `PUBLISHING.md` for GitHub Pages layout and notebook-link configuration.

## Robot models: actual geometry, explicit teaching controls

Run once when reaching the embodiment chapters:

```bash
python lab/setup_robot_assets.py
```

This uses Git and downloads about 151 MiB from pinned upstream model subtrees.
The code preserves licenses and source commit markers. It does not require
ROS, Isaac Sim, SAPIEN, or robosuite for these new native MuJoCo reaching tasks.
The download was tested separately from the local pre-fetched assets.

| Model | Native source | Course action interface | Boundary |
|---|---|---|---|
| Franka Panda | MuJoCo Menagerie | 7 normalized joint-target increments | Native gripper [0,255] is held separately; reaching is not grasping |
| AgileX ALOHA | Official AgileX legacy MJCF | 12 arm increments, fingers held | Fixed chassis; not DeepMind ALOHA 2; original scene contains objects |
| Galaxea R1 Pro | Official ManipSim URDF → passive MuJoCo import | 18 torso/arm increments with course PD torque control | Source chassis fixed; controls are not factory calibrated |
| R1 Pro + ideal planar base | Same upper-body geometry, course extension | 21 commands | Adds x/y/yaw joints; not wheel steering, rolling/slip, or terrain validation |

Read `PROVENANCE.md` and `assets_manifest.json`. The vendor files are fetched
separately rather than inflating the learning download. The course does not
rename Tiago or ALOHA 2 as one of the requested embodiments.

The native models use different controls. In particular, Panda arm targets are
radians while its shared gripper control is [0,255]; AgileX arm targets are
radians and finger slides are metres. Course normalized commands are a separate
documented wrapper. Use named joint indexing; background objects add qpos entries.

The new reaching tasks intentionally isolate RL coordination and control contracts.
They do not establish solved grasping, bimanual contact manipulation, navigation,
or real-robot safety. Existing robosuite Lift/TwoArmLift companion notebooks remain
available through the earlier simulator page, labeled with their original robots.

## The code is visible

`lab/rl_core.py` contains original returns, Bellman backup, replay, DQN update,
REINFORCE, AC/GAE, PPO, TD3 and SAC code. `lab/embodiment_env.py` defines actual
MuJoCo tasks; `lab/physics_lab.py` supplies the isolated residual/reset wrappers.
Some notebooks include editable function definitions and explicitly reconnect
them to the trainer. Editing a local function does not silently change an imported
module—follow that wiring cell.

The core implementations prioritize readability, correct boundary masks, detached
targets, consistent action coordinates, and inspectable gradients. They are not
drop-in production robot controllers or optimized training frameworks.

### Longer experiments

```bash
python lab/rl_core.py --algo PPO --env Reacher-v5 --steps 100000 --seed 7 --device cpu --out my-runs
python lab/rl_core.py --algo SAC --env Reacher-v5 --steps 100000 --seed 7 --device cpu --out my-runs
```

In a robot notebook, increase the `train_continuous` step budget after the short
correctness run. Always close environments when done. Choose identical transitions,
three or more training seeds, untouched evaluation resets, and a fixed success
definition for comparisons. CPU and GPU wall time should be measured, not inferred.

## Research implementation status

| Topic | What the curriculum implements | What it does not claim |
|---|---|---|
| PPO / TD3 / SAC | Original small-network update code and real environment execution | Multi-seed superiority or factory-tuned controllers |
| GRPO | Group arithmetic and a differentiable grouped bandit-ratio exercise | Validated continuous-control GRPO trainer |
| QF3 | Flow-policy explanation and isolated guidance clipping | Full flow actor, critic recipe and paper reproduction |
| SGS / Balanced Data Diet | Finite seeded robot reset success histories and weighting | Full paper benchmark/configuration protocol |
| ACCRUE | Continual-learning reasoning and retention evaluation design | Robot foundation actor/critic continual-training system |
| SARM2 / RARM | Progress/value versus immediate reward; gating and shaping analysis | Learned visual reward models or their annotation pipelines |
| ResFiT | Bounded residual composition and actual residual SAC updates | Frozen BC action chunks, demonstrations and exact training recipe |
| EXPO-FT / Real-Time EXPO-FT | Critic-selection and timing/queue reasoning | VLA edits, interventions and real-time chunk implementation |
| FastTD3 / FlashSAC | TD3/SAC foundations and update/throughput accounting | Distributional critics, GPU-scale systems and paper results |

## Verification

All 18 notebooks' default code cells were executed in-process with IPython on Linux Python 3.12 CPU.
Every notebook includes actual mathematical outputs and an original figure.
The validation includes actual Gymnasium transitions and optimizer updates;
vendor models compile, step, and render; custom reaching environments have seeded
resets, finite observations/actions, and explicit stopping semantics.
See `execution.json` and `validation.json` for exact checks and limits.

Browser DOM logic and math rendering are checked separately. No supported browser
controller was available, so visual/layout QA is not claimed. A separate Jupyter
kernel launch was blocked by this host's socket restrictions; the code cells and
rich outputs were validated through in-process IPython instead. Mac-native and
Colab CUDA execution were not available. Short integration runs do not prove
learning convergence or research-level generalization.
