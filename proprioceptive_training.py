
# !!!!!   NOT REVIEWED   !!!!!



import functools
from datetime import datetime
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from ml_collections import config_dict
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import losses as ppo_losses
import custom_ppo_train
from utils import render_video_during_training, evaluate_policy


# 1. Define environment and reward config (proprioceptive only)
def reward_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,
        Kp=35.0,
        Kd=0.5,
        action_repeat=1,
        action_scale=1.0,
        history_len=1,
        soft_joint_pos_limit_factor=0.95,
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
                linvel=0.1,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                torso_height=3.0,
                tracking_lin_vel=5.0,
                tracking_ang_vel=2.5,
                lin_vel_z=-0.5,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                dof_pos_limits=-1.0,
                pose=0.5,
                termination=-1.0,
                stand_still=-1.0,
                torques=-0.0002,
                action_rate=-0.01,
                energy=-0.001,
                feet_clearance=-0.3,
                feet_slip=-0.1,
                feet_air_time=0.1,
            ),
            tracking_sigma=0.25,
            max_foot_height=0.20,
            desired_foot_air_time=0.15,
            desired_torso_height=0.36,
        ),
        pert_config=config_dict.create(
            enable=False,
            velocity_kick=[0.0, 3.0],
            kick_durations=[0.05, 0.2],
            kick_wait_times=[1.0, 3.0],
        ),
        command_config=config_dict.create(
            a=[1.5, 0.8, 1.2],
            b=[0.9, 0.25, 0.5],
        ),
        impl="jax",
        nconmax=4 * 8192,
        njmax=40,
    )

# 2. Load environment

env_name = 'Go1JoystickRoughTerrain'  # Or your baseline env
env = registry.load(env_name, config=reward_config())
key = jax.random.PRNGKey(15)
randomizer = registry.get_domain_randomizer(env_name)

# 3. PPO parameters
ppo_params = locomotion_params.brax_ppo_config(env_name)
ppo_training_params = dict(ppo_params)
ppo_training_params["num_evals"] = 25
ppo_training_params["num_timesteps"] = 200_000_000
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory
    )

# 4. Progress and policy callback
x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]
current_policy = None
def progress(num_steps, metrics):
    from IPython.display import clear_output, display
    clear_output(wait=True)
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])
    plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"Proprioceptive Training: reward={y_data[-1]:.3f}")
    plt.errorbar(x_data, y_data, yerr=y_dataerr, color="red")
    display(plt.gcf())
    if current_policy is not None:
        eval_env_for_video = registry.load(env_name, config=reward_config())
        jit_reset = jax.jit(eval_env_for_video.reset)
        jit_step = jax.jit(eval_env_for_video.step)
        render_video_during_training(current_policy, num_steps, jit_step, jit_reset, reward_config(), eval_env_for_video)

def policy_params_callback(_, make_policy_fn, params):
    global current_policy
    current_policy = make_policy_fn(params, deterministic=True)

# 5. Training function
train_fn = functools.partial(
    custom_ppo_train.train,
    **ppo_training_params,
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress,
    policy_params_fn=policy_params_callback,
)

if __name__ == "__main__":
    print("Starting proprioceptive-only training...")
    make_policy, params, _ = train_fn(
        environment=env,
        eval_env=registry.load(env_name, config=reward_config()),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        compute_custom_ppo_loss_fn=ppo_losses.compute_ppo_loss
    )
    print("Training completed.")
    # Optionally evaluate
    eval_env = registry.load(env_name, config=reward_config())
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference_fn = jax.jit(make_policy(params, deterministic=True))
    print("Evaluating policy...")
    evaluate_policy(
        eval_env,
        jit_inference_fn,
        jit_step,
        jit_reset,
        reward_config(),
        eval_env,
        [0.0, 0.0],
        [0.05, 0.2],
    )
