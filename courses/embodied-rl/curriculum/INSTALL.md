# Install a workbench, then run one measured experiment

The recommended track combines the original readable PyTorch code with an optional
Stable Baselines3 baseline. The robot tasks use Gymnasium and native MuJoCo.
LeRobot and RLinf are separate advanced workbenches, not dependencies of the first
18 lessons. Commands below run from the extracted course folder.

## 1. Mac or Linux CPU: course + Stable Baselines3

Install Python 3.11 or 3.12 with a package manager you already use. Apple Silicon
is a suitable CPU learning machine. Use the interpreter explicitly so a system
Python does not accidentally receive these packages.

```bash
python3 scripts/install.py --python python3.11 --profile sb3 --dry-run
python3 scripts/install.py --python python3.11 --profile sb3
.venv/bin/python -m jupyter lab notebooks/00_probability.ipynb
```

The installer creates `.venv`, installs course requirements plus SB3 2.9.0, steps
an actual Reacher environment, constructs an SB3 SAC model, and writes the exact
resolved dependency list to `.venv/course-freeze.txt`. Use `--profile core` to omit
SB3. Existing environments are reused; choose another `--venv` path for a clean one.
The dependency ranges allow supported architecture wheels; the freeze file records
your actual installation. This is not a claim that every platform resolves identically.

Manual equivalent:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-sb3.txt
python scripts/doctor.py --profile sb3
```

Use native rendering on macOS. On a headless Linux machine, set `MUJOCO_GL=egl`
before importing MuJoCo. Rendering needs an available GL implementation; the
non-rendering doctor check does not certify your desktop display setup.

## 2. Colab: open the selected chapter directly

Every chapter's button uses Colab's `/github/OWNER/REPO/blob/REF/PATH.ipynb`
format. The public review branch supplies the actual notebook. The first setup
cell carries a compressed snapshot of the editable course modules and restores
them into the runtime if needed. There is no ZIP-upload step or private-site login
inside the notebook. Inspect the setup cell before running, as with any notebook.

The cell preserves Colab's installed PyTorch, installs small course essentials,
and defaults to CPU. Set `USE_GPU=True` only after allocating a GPU runtime. The
first robot lab separately downloads about 151 MiB of pinned model assets.
To try the library track in an existing Colab session:

```python
%pip install stable-baselines3==2.9.0
from sb3_baseline import run
report = run('SAC', 'Reacher-v5', steps=10_000, seed=7)
```

Colab may require a Google sign-in and a third-party-notebook confirmation. GPU
allocation varies. Save modified notebooks and run outputs before the session ends.

## 3. Run a library baseline on the same task

```bash
.venv/bin/python lab/sb3_baseline.py --algo SAC --env Reacher-v5 --steps 10000 --seed 7
.venv/bin/python lab/sb3_baseline.py --algo PPO --env Reacher-v5 --steps 10000 --seed 7
.venv/bin/python lab/setup_robot_assets.py
.venv/bin/python lab/sb3_baseline.py --algo SAC --env franka --steps 10000 --seed 7
```

`--env agilex`, `r1-pro`, and `r1-pro-planar` select the other course tasks.
Use `--algo DQN --env CartPole-v1` for the discrete-action baseline.
AC's nearest provided SB3 comparator is A2C, not an identical implementation.
The runner also exposes TD3 and DDPG. It saves a model, Monitor CSV and JSON with
requested/actual transitions, versions, seed and separate deterministic evaluation.
On-policy collection may round up to a complete rollout; compare actual counts.
The Reacher task has no course success definition, so its report uses returns.
Robot reports use the environment's held-success condition. A short run is an
integration check. Use multiple training seeds and matched budgets for conclusions.

## 4. LeRobot: dataset and pretrained-policy work

Use this when a lesson needs demonstration datasets, policy training, or a real
robot adapter. It does not automatically convert the course MuJoCo environment
into a supported LeRobot robot. A Galaxea interface and calibrated action mapping
remain integration work.

```bash
bash scripts/install-lerobot.sh
```

This source install is pinned to commit
`297774fca4e3ad0aa34e7425fbf937111637e77b` and uses a separate Python3.12
environment with the training extra. Current upstream documentation calls for
Python>=3.12 and PyTorch>=2.10. Video decoding may need an ffmpeg installation
appropriate to TorchCodec; follow the official platform instructions. The course
does not install hardware drivers, calibrate a robot or download policy weights.

## 5. RLinf: advanced Linux GPU server work

Use RLinf when you are ready to distribute VLA/environment/rollout workloads.
The supplied script targets Linux and the upstream OpenVLA + ManiSkill/LIBERO
installation combination, pinned to commit
`7e90641c4135229db6d3426950cdb45cee6259a6`. It is not the Mac CPU course path.

```bash
bash scripts/install-rlinf.sh
```

Read the upstream hardware requirements first. A multi-GPU cluster or different
simulator needs its own documented model/environment configuration; a default
Colab GPU is not a universal substitute. This script installs the example
workbench, not a Galaxea whole-body experiment. No RLinf cluster run or LeRobot
source installation was executed for this release.

## Evidence and primary setup references

Verified here: Linux CPU lesson code, real robot model stepping, the doctor check,
and short SB3 SAC/PPO runs. Mac-native and Colab CUDA execution remain untested.
See `execution.json`, `validation.json` and `sb3-validation.json` for exact evidence.

- https://stable-baselines3.readthedocs.io/en/master/guide/install.html
- https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
- https://huggingface.co/docs/lerobot/installation
- https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html
- https://github.com/googlecolab/open_in_colab
