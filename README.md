# Perception-Based RL for Quadruped Locomotion

This repository contains the final project for **Robot Learning and Embodied AI (DD2600)** at KTH Royal Institute of Technology. The project extends a MuJoCo/Brax reinforcement-learning lab for the Go1 quadruped with perception-based locomotion policies.

**Group members:**
- Hugo Dezerto
- Lucas Lind
- Jennifer Lundström

## Project Overview

The project compares three policy setups for simulated quadruped locomotion:

- **Proprioceptive baseline:** a PPO policy trained from robot-state observations only.
- **Exteroceptive policy:** a PPO policy that adds terrain-height observations from ray-based perception.
- **Teacher-student setup:** a privileged teacher policy and a student policy trained with imitation loss.

The custom environment adds wall/terrain randomization, exteroceptive scanning, curriculum-learning support, and evaluation utilities for velocity-tracking behavior.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `custom_env.py` | Custom Go1 joystick environment with wall randomization and exteroceptive observations. |
| `custom_ppo_train.py` | Single-device PPO training loop adapted for the project, including teacher-student imitation mode. |
| `utils.py` | Evaluation, plotting, and video-rendering helpers. |
| `visualize_custom_env.py` | Standalone visualization script for the custom environment and height scanner. |
| `custom_env.xml` | Main MuJoCo XML model. |
| `custom_env_debug_wall.xml` | Debug MuJoCo XML model used for wall/raycast visualization. |
| `proprioceptive.ipynb` | Training and evaluation notebook for the proprioceptive baseline. |
| `exteroceptive.ipynb` | Training and evaluation notebook for the exteroceptive policy. |
| `teacher_student.ipynb` | Teacher-student training and evaluation notebook. |
| `plot_metrics.ipynb` | Notebook for plotting generated evaluation metrics. |
| `policies/` | Curated trained policy snapshots serialized with `dill`. |
| `eval_videos/` | Curated evaluation videos grouped by policy type. |

See [`docs/artifacts.md`](docs/artifacts.md) for details about the saved policies, videos, and generated metric files.

## Setup

This project builds on the original DD2600 RL lab environment. Start by following the setup instructions in the [upstream repository](https://github.com/finnBsch/eai2025_rl_final).

For curriculum learning and full environment resets, install the project-specific simulation branches inside the same virtual environment:

```bash
source venv_rl/bin/activate
python3 -m pip uninstall playground
python3 -m pip install git+https://github.com/finnBsch/mujoco_playground.git@full_reset
python3 -m pip uninstall mujoco-mjx
python3 -m pip install git+https://github.com/finnBsch/mujoco.git@lab#subdirectory=mjx
```

The notebooks assume the upstream lab dependencies are available, including JAX, MuJoCo, Brax, `mujoco_playground`, `dill`, `mediapy`, OpenCV, and Matplotlib.

## Usage

The notebooks are the main entry points:

1. Run `proprioceptive.ipynb` for the proprioceptive PPO baseline.
2. Run `exteroceptive.ipynb` for the terrain-aware exteroceptive policy.
3. Run `teacher_student.ipynb` for privileged teacher training and student imitation training.
4. Run `plot_metrics.ipynb` after generating evaluation metrics.

Training runs are configured for long experiments by default, for example `NUM_TIMESTEPS = 200_000_000` in the notebooks. For quick smoke tests, reduce `NUM_TIMESTEPS` and `NUM_EVALS` before running a notebook.

To inspect the environment and height-scanner visualization without running a full training job:

```bash
python visualize_custom_env.py
```

## Artifacts and Metrics

The repository includes curated trained policies and evaluation videos so the project can be inspected without rerunning the full training pipeline.

- Saved policies live in `policies/`.
- Evaluation videos live in `eval_videos/<policy>/` with names like `teacher_velocity_case_0.mp4`.
- Metric files are generated outputs and are not committed by default. `plot_metrics.ipynb` expects files such as `metrics/proprioceptive_metrics.dill`, `metrics/exteroceptive_metrics.dill`, and `metrics/student_metrics.dill` after evaluation has been run.

The policy files use `dill`, which can execute code while loading serialized objects. Only load these files from a trusted copy of the repository.

## Project Status

This repository is a research project snapshot with curated trained policies and evaluation videos included for review. The notebooks document the experimental workflow, while the source files contain the reusable environment, training, evaluation, and visualization code.

## Credits

The project builds on the DD2600 RL lab and adapts components from Brax, MuJoCo/MJX, and `mujoco_playground`.
