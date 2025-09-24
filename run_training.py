#!/usr/bin/env python3
"""Visualization script for Go1 with height scanner."""
import os
# Tell XLA to use Triton GEMM
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags
os.environ['MUJOCO_GL'] = 'egl'

# IMPORTS FROM lab1_C
from datetime import datetime
import os
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import functools
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import wrapper
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import losses as ppo_losses
import mujoco
import jax
import jax.numpy as jp
import cv2
import custom_ppo_train
from utils import render_video_during_training, evaluate_policy
import mediapy as media

# IMPORTS FROM VISUALIZE
import jax
import jax.numpy as jp
import numpy as np
import mujoco
import cv2
from PIL import Image, ImageDraw, ImageFont
from custom_env import Joystick, default_config

# Set up visualization options
scene_option = mujoco.MjvOption()
scene_option.geomgroup[2] = True   # Show visual geoms
scene_option.geomgroup[3] = False  # Hide collision geoms
scene_option.geomgroup[5] = True   # Show sites (including height scanner visualization)
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True  # Show contact points
scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True

# We will no longer use the registry, but directly load our custom XML and model.
xml_path = 'custom_env.xml'
env = Joystick(xml_path=xml_path, config=default_config())

# NOTE: For the visualization test, we manually set init_q z position high to avoid collisions with walls
# I assume we do not want to do this anymore
# env._init_q = env._init_q.at[2].set(1.0)

# JIT compile the functions for speed
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_terrain_height = jax.jit(env._get_torso_terrain_height)
jit_actual_terrain_height = jax.jit(env._get_terrain_height)

# Initialize
key = jax.random.PRNGKey(15)

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

current_policy = None

# Loading the environment (not the one for debugging)
xml_path = "custom_env.xml"

# NOTE: Since we do not use registry anymore we change it for loading with Joystick:
# eval_env_for_video = registry.load(env_name, config=env_cfg)      # Old load with registry
eval_env_for_video = Joystick(xml_path=xml_path, config=default_config())   # New load using Joystick
jit_reset = jax.jit(eval_env_for_video.reset)
jit_step = jax.jit(eval_env_for_video.step)

def progress(num_steps, metrics):
    # NOTE: PReviously we used clear_putput() which is Jupyter
    # clear_output(wait=True)       # Old command
    plt.clf()                       # New command

    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])

    plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"Challenging Terrain Training: reward={y_data[-1]:.3f}")
    plt.errorbar(x_data, y_data, yerr=y_dataerr, color="red")
    
    # NOTE: We previosuly used display which is a Jupyter command, we change it to plt.pause()
    # display(plt.gcf())    # Jupyter command to display
    plt.pause(0.001)        # for live plotting in script
        
    # Render video if we have a current policy
    if current_policy is not None:
        render_video_during_training(current_policy, num_steps, jit_step, jit_reset, env.config, eval_env_for_video)

def render_trained_policy(policy, env, key, num_steps=200, gif_path="trained_rollout.gif"):
    """Roll out the trained policy and save GIF."""
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    state = jit_reset(key)
    rollout = []

    for t in range(num_steps):
        rollout.append(state)
        action = policy(state.obs)
        state = jit_step(state, action)

    frames = env.render(
        rollout,
        camera="track",
        width=480,
        height=360,
    )

    pil_images = [Image.fromarray(f) for f in frames]
    pil_images[0].save(
        gif_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=50,
        loop=0
    )
    print(f"✓ GIF saved to {gif_path}")

def main():
    # PPO training parameters - you may want to tune these for challenging terrain
    ppo_params = locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain")  # any valid name
    #ppo_params = locomotion_params.brax_ppo_config(env_name)
    ppo_training_params = dict(ppo_params)
    
    ppo_training_params["num_evals"] = 5 # Reduce for final training for less feedback.
    ppo_training_params["num_timesteps"] = 25000000  # Total number of training steps

    network_factory = ppo_networks.make_ppo_networks

    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )

    print("Training parameters:")
    print(ppo_training_params)

    # Create a policy parameters callback to capture the current policy
    def policy_params_callback(_, make_policy_fn, params):
        global current_policy
        current_policy = make_policy_fn(params, deterministic=True)
        
    train_fn = functools.partial(
            custom_ppo_train.train,
            **ppo_training_params,
            network_factory=network_factory,
            #randomization_fn=randomizer,
            progress_fn=progress,
            policy_params_fn=policy_params_callback,
    )

    print("Starting training on challenging terrain...")
    make_policy, params, _ = train_fn(
        environment=env,  # your main training env
        eval_env=Joystick(xml_path=xml_path, config=default_config()),  # evaluation env
        wrap_env_fn=wrapper.wrap_for_brax_training,  # environment wrapper for Brax
        compute_custom_ppo_loss_fn=ppo_losses.compute_ppo_loss  # custom PPO loss
    )
    print("Training completed.")

    render_trained_policy(make_policy, eval_env, key, num_steps=200, gif_path="trained_policy.gif")


if __name__ == "__main__":
    main()