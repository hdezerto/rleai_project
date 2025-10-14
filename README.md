# Project: Extend RL Lab with Perception

This repository contains code for the final project in the course **Robot Learning and Embodied AI (DD2600)** at KTH Royal Institute of Technology.

**Group members:**
- Hugo Dezerto
- Lucas Lind
- Jennifer Lundström

## Project Overview
This project extends a reinforcement learning (RL) lab by adding perception-based policies for quadruped locomotion in simulation. It explores curriculum learning, exteroceptive and proprioceptive policies, and teacher-student setups using MuJoCo and PPO.

## Setup Instructions
1. Follow the environment setup in the [original repository](https://github.com/finnBsch/eai2025_rl_final).
2. To enable curriculum learning (full environment reset), update the simulation packages:

```bash
source venv_rl/bin/activate
python3 -m pip uninstall playground
python3 -m pip install git+https://github.com/finnBsch/mujoco_playground.git@full_reset
python3 -m pip uninstall mujoco-mjx
python3 -m pip install git+https://github.com/finnBsch/mujoco.git@lab#subdirectory=mjx
```

## Usage
- **Proprioceptive baseline:** See `proprioceptive.ipynb`
- **Exteroceptive policy:** See `exteroceptive.ipynb`
- **Teacher-student setup:** See `teacher_student.ipynb`


## Folder Structure
- `policies/` – Saved policies
- `eval_videos/` – Evaluation videos
- `metrics/` – Saved evaluation metrics
