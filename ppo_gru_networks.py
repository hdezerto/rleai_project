import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import losses as ppo_losses
from brax.training import types
import distrax
from types import SimpleNamespace
from brax.training.acme import running_statistics


class GRUPolicyMLP(nn.Module):
    obs_size: int
    action_size: int
    hidden_layer_sizes: tuple = (512, 256, 128)
    hidden_size: int = 128  # choose the GRU state size

    @nn.compact
    def __call__(self, x, carry):
        new_carry, gru_out = nn.GRUCell(features=self.hidden_size)(carry, x)
        h = nn.relu(nn.Dense(self.hidden_layer_sizes[0])(gru_out))
        h = nn.relu(nn.Dense(self.hidden_layer_sizes[1])(h))
        h = nn.relu(nn.Dense(self.hidden_layer_sizes[2])(h))
        mean = nn.Dense(self.action_size)(h)
        log_std = self.param('log_std', lambda k: jnp.zeros((self.action_size,)))
        return mean, log_std, new_carry


def make_ppo_gru_networks(
    obs_shape,
    action_size,
    preprocess_observations_fn=None,
    hidden_layer_sizes=(512, 256, 128)  # Default matches your config
):
    if isinstance(obs_shape, dict): # Handle dict observation spaces
        obs_size = sum([int(np.prod(v)) for v in obs_shape.values()])
    else: # Handle array observation spaces
        obs_size = int(np.prod(obs_shape))
    # Custom policy network that uses a GRU followed by MLP layers
    policy_net = GRUPolicyMLP(
            obs_size=obs_size,
            action_size=action_size,
            hidden_layer_sizes=hidden_layer_sizes,
        )
    
    # Standard MLP value network
    ppo_mlp_networks = ppo_networks.make_ppo_networks(
    obs_shape,
    action_size,
    preprocess_observations_fn=preprocess_student_dict,
    )
    value_net = ppo_mlp_networks.value_network
    parametric_action_distribution = ppo_mlp_networks.parametric_action_distribution

    return ppo_networks.PPONetworks(
        policy_network=policy_net,
        value_network=value_net,
        parametric_action_distribution=parametric_action_distribution,
    )


def preprocess_student_dict(obs, normalizer_params):
    """
    Preprocesses student observations to a flat normalized array.
    - Accepts obs as either a dict containing 'student_state' or a flat array.
    - Works with normalizer_params that are either a flat RunningStatisticsState
      or a dict containing a 'student_state' entry.
    Returns a normalized array suitable for value/policy networks expecting
    shape (student_size,).
    """
    if isinstance(obs, dict):
        arr = obs.get("student_state", None)
        if arr is None:
            # If dict without student_state, try to concatenate leaves in sorted order
            keys = sorted(list(obs.keys()))
            arr = jnp.concatenate([obs[k] for k in keys], axis=-1)
    else:
        arr = obs

    # Select appropriate normalizer params shape
    ns = normalizer_params
    if isinstance(normalizer_params, dict):
        ns = normalizer_params.get("student_state", normalizer_params)

    return running_statistics.normalize(arr, ns)


def make_gru_policy(policy_network, value_network, params, preprocess_observations_fn, normalizer_params, deterministic=False):
    def policy_fn(obs, carry, key):
        proc_obs = preprocess_observations_fn(obs, normalizer_params)
        mean, log_std, new_carry = policy_network.apply(params.policy, proc_obs, carry)
        std = jnp.exp(log_std)
        dist = distrax.Normal(loc=mean, scale=std)
        if deterministic:
            action = mean
        else:
            action = dist.sample(seed=key)
        log_prob = dist.log_prob(action).sum(-1)
        # For value network, normalize only student_state dict
        value = value_network.apply(normalizer_params, params.value, preprocess_student_dict({"student_state": obs["student_state"]}, normalizer_params))
        entropy = dist.entropy().sum(-1)
        extras = {'log_prob': log_prob, 'value': value, 'entropy': entropy, 'mean_action': mean}
        return action, new_carry, extras
    policy_fn.value = lambda obs: value_network.apply(
        normalizer_params,
        params.value,
        preprocess_student_dict({"student_state": obs["student_state"]}, normalizer_params),
    )
    def init_carry(batch_size, key):
        # Explicitly return zeros of shape (batch_size, hidden_size) to avoid API ambiguities
        del key
        return jnp.zeros((batch_size, policy_network.hidden_size), dtype=jnp.float32)
    policy_fn.init_carry = init_carry
    return policy_fn


def make_gru_policy_fn(policy_network, value_network, preprocess_observations_fn, ):
    """
    Returns a function that takes (params, deterministic) and returns a GRU policy callable.
    This matches the make_policy signature used elsewhere.
    """
    def policy_factory(params, deterministic=False):
        # params is a tuple: (normalizer_params, policy_params, value_params)
        normalizer_params, policy_params, value_params = params
        param_obj = SimpleNamespace(policy=policy_params, value=value_params)
        return make_gru_policy(
            policy_network=policy_network,
            value_network=value_network,
            params=param_obj,
            preprocess_observations_fn=preprocess_observations_fn,
            normalizer_params=normalizer_params,
            deterministic=deterministic,
        )
    return policy_factory


def generate_gru_unroll(env, state, policy, carry, key, unroll_length, teacher_policy_fn=None, extra_fields=()):
    batch_size = carry.shape[0]

    def step_fn(carry_tuple, _):
        state, carry, key = carry_tuple
        key, subkey = jax.random.split(key)
        obs = state.obs
        action, new_carry, policy_extras = policy(obs, carry, subkey)
        nstate = env.step(state, action)
        reset_carry = policy.init_carry(batch_size, subkey)
        new_carry = jnp.where(nstate.done[:, None], reset_carry, new_carry)

        extras = dict(policy_extras)
        # Safely gather optional fields; default to None if not present on State
        state_extras = {field: getattr(nstate, field, None) for field in extra_fields}
        extras['state_extras'] = state_extras

        if teacher_policy_fn is not None:
            teacher_out = teacher_policy_fn(obs, subkey)
            teacher_action = teacher_out[0] if isinstance(teacher_out, tuple) else teacher_out
            extras['teacher_action'] = teacher_action

        discount = getattr(nstate, 'discount', 1.0 - nstate.done)
        out = types.Transition(
            observation=obs,
            action=action,
            reward=nstate.reward,
            discount=discount,
            next_observation=nstate.obs,
            extras=extras,
        )
        return (nstate, new_carry, key), out

    (final_state, final_carry, _), traj = jax.lax.scan(
        step_fn,
        (state, carry, key),
        None,
        length=unroll_length,
    )
    last_obs = final_state.obs
    bootstrap_value = policy.value(last_obs)  # [num_envs]
    extras = dict(traj.extras)
    # Add singleton time-like dimension so it participates in time/batch swapping
    extras['bootstrap_value'] = bootstrap_value[:, None]
    # types.Transition is a NamedTuple; use _replace to update the extras field
    traj = traj._replace(extras=extras)
    return final_state, final_carry, traj



def compute_student_ppo_imitation_loss(
    params,
    normalizer_params,
    data,
    key,
    *,
    ppo_network,
    lambda_imitation=0.1,
    **kwargs,
):
    # Standard PPO loss
    ppo_loss, metrics = ppo_losses.compute_ppo_loss(
        params,
        normalizer_params,
        data,
        key,
        ppo_network=ppo_network,
        **kwargs,
    )
    # Imitation loss (MSE between student and teacher actions)
    imitation_loss = jnp.mean((data.action - data.extras['teacher_action']) ** 2)
    total_loss = ppo_loss + lambda_imitation * imitation_loss
    metrics['imitation_loss'] = imitation_loss
    return total_loss, metrics



def gru_evaluator(
    env,
    policy_with_carry,
    params,
    normalizer_params,
    preprocess_observations_fn,
    episode_length,
    key,
    num_eval_envs=1,
):
    """
    Evaluates a GRU policy with carry, returns metrics similar to Brax's Evaluator.
    """
    # Reset envs
    reset_keys = jax.random.split(key, num_eval_envs)
    state = env.reset(reset_keys)
    carry = policy_with_carry.init_carry(num_eval_envs, key)
    total_rewards = jnp.zeros(num_eval_envs)
    lengths = jnp.zeros(num_eval_envs)
    dones = jnp.zeros(num_eval_envs, dtype=bool)

    def step_fn(carry_tuple, _):
        state, carry, total_rewards, lengths, dones, key = carry_tuple
        key, subkey = jax.random.split(key)
        obs = state.obs
        action, new_carry, _ = policy_with_carry(obs, carry, subkey)
        nstate = env.step(state, action)
        reward = nstate.reward
        done = nstate.done
        # Only accumulate reward/length for unfinished episodes
        total_rewards = total_rewards + reward * (~dones)
        lengths = lengths + (~dones)
        dones = jnp.logical_or(dones, done)
        return (nstate, new_carry, total_rewards, lengths, dones, key), None

    (final_state, final_carry, total_rewards, lengths, dones, _), _ = jax.lax.scan(
        step_fn,
        (state, carry, total_rewards, lengths, dones, key),
        None,
        length=episode_length,
    )

    # Compute metrics as Brax does
    metrics = {
        "eval/episode_reward": jnp.mean(total_rewards),
        "eval/episode_length": jnp.mean(lengths),
        "eval/episode_reward_std": jnp.std(total_rewards),
        "eval/episode_length_std": jnp.std(lengths),
    }
    return metrics