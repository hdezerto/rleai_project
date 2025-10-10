# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for Go1."""

from logging import info
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import ray
from mujoco.mjx._src import math
import numpy as np
from jax.experimental import io_callback

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import go1_constants as consts
from mujoco_playground._src import collision


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02, # Control timestep (for action updates)
        sim_dt=0.004, # Simulation timestep
        episode_length=1000, # Number of control steps per episode
        Kp=35.0, # Proportional gain
        Kd=0.5, # Derivative gain
        action_repeat=1, # Number of simulation steps per control step
        action_scale=1.0, 
        history_len=1, # Length of history for observations (1 = no history)
        soft_joint_pos_limit_factor=0.95,
        # Observation noise configuration
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
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
                torso_height=-0.0,  # Adjust magnitude
                # Tracking.
                tracking_lin_vel=1.0,
                tracking_ang_vel=0.5,

                # Base reward.
                lin_vel_z=-0.5,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                # Other.
                dof_pos_limits=-1.0, 
                pose=0.5,
                # Other.
                termination=-1.0,
                stand_still=-1.0, 
                # Regularization.
                torques=-0.0002,
                action_rate=-0.01,
                energy=-0.001, 
                # Feet.
                feet_clearance=-0.2,
                feet_slip=-0.1,
                feet_air_time=0.1,
                knee_collisions=-1.0,     # Add term for knee collisions   
            ),
            tracking_sigma=0.25,
            max_foot_height=0.15,    
            desired_foot_air_time=0.15, 
            desired_torso_height=0.36,   
        ),
        # For adding random pushes to the robot (for robustness)
        pert_config=config_dict.create(
            enable=False,
            velocity_kick=[0.0, 3.0],
            kick_durations=[0.05, 0.2],
            kick_wait_times=[1.0, 3.0],
        ),
        # Command sampling configuration
        command_config=config_dict.create(
            # Uniform distribution for command amplitude.
            a=[1.5, 0.8, 1.2],
            # Probability of not zeroing out new command.
            b=[0.9, 0.25, 0.5],
        ),
        impl="jax",
        nconmax=4 * 8192,
        njmax=40,
    )


def parse_binary_data(binary_data):
    """
    Parse binary data containing null-terminated strings
    
    Args:
        binary_data: bytes object or string representation of bytes
    
    Returns:
        list: List of extracted strings
    """
    
    # If input is a string representation of bytes, convert it
    if isinstance(binary_data, str) and binary_data.startswith("b'"):
        # Remove b' prefix and ' suffix, then decode escape sequences
        binary_str = binary_data[2:-1]
        binary_data = bytes(binary_str, 'utf-8').decode('unicode_escape').encode('latin-1')
    
    # Split by null bytes and filter out empty strings
    strings = binary_data.split(b'\x00')
    parsed_strings = [s.decode('utf-8', errors='ignore') for s in strings if s]
    
    return parsed_strings



class Joystick(go1_base.Go1Env):
    """Track a joystick command."""

    def __init__(
        self,
        xml_path: str = None, 
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        total_steps: Optional[int] = None,  # New parameter for curriculum learning
        steps_per_reset: Optional[int] = None,  # Only used for visualizing the environment
        curriculum_learning: bool = False,
        exteroceptive: bool = False,  # New flag to control exteroceptive state
    ):
        if xml_path is None:
            raise ValueError("xml_path must be provided for Joystick environment.")
        config.nconmax = 100 * 8192
        config.njmax = 12 + 100 * 4
        # Calls the parent class (Go1Env) constructor with the provided arguments to initialize the base environment.
        super().__init__(
            xml_path=xml_path,
            config=config,
            config_overrides=config_overrides,
        )
        self._reset_count = 0  # Host-side counter for number of resets (not trusted inside JIT)
        self._total_steps = total_steps  # Store total_steps for curriculum learning
        if steps_per_reset is None:
            self._episode_length = self._config.episode_length
        else:
            self._episode_length = steps_per_reset
        self._curriculum_learning = curriculum_learning  # If True, disables curriculum learning
        self._exteroceptive = exteroceptive  # Store exteroceptive flag
        self._post_init()  # Custom post-init to set up additional attributes


    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:19])  # Only robot joints

        # Note: First joint is freejoint, followed by 12 robot joints, then wall joints.
        self._lowers, self._uppers = self.mj_model.jnt_range[1:13].T  # Only robot joints
        self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

        self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in consts.FEET_SITES]
        )
        self._knee_geom_id = np.array(
            [self._mj_model.geom(name).id for name in consts.KNEE_GEOMS]
        )
        self._floor_geom_id = self._mj_model.geom("floor").id

        # NOTE: We identify all the wall geoms by their name containing "wall". This allows us to access them later.
        wall_name_substr = "wallgeom"
        self._wall_geom_ids = np.array([
            self._mj_model.geom(name).id
            for name in parse_binary_data(self._mj_model.names)
            if wall_name_substr in name
        ])
        wall_name_substr = "wallbody"
        self._wall_body_ids = np.array([
            self._mj_model.body(name).id
            for name in parse_binary_data(self._mj_model.names)
            if wall_name_substr in name
        ])

        self._feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
        )

        foot_linvel_sensor_adr = []
        for site in consts.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        self._cmd_a = jp.array(self._config.command_config.a)
        self._cmd_b = jp.array(self._config.command_config.b)

        self._robot_body_ids = []    
        # NOTE: Here we get all the robot body ids, to exclude them when raycasting.
        # Get all bodies that are part of the robot subtree
        # This assumes the robot is a connected kinematic chain starting from ROOT_BODY
        for body_id in range(self._mjx_model.nbody):
            # Check if this body is part of the robot by checking if it's in the subtree of ROOT_BODY
            current_body_id = body_id
            while current_body_id != 0:  # 0 is typically the world body
                parent_id = self._mjx_model.body_parentid[current_body_id]
                if parent_id == self._torso_body_id or current_body_id == self._torso_body_id:
                    self._robot_body_ids.append(body_id)
                    break
                current_body_id = parent_id
                
    # NOTE: Add property to allow usage outside of class
    @property
    def exteroceptive(self):
        return self._exteroceptive

    # ========================== CHECK: Wall control methods ==========================
    # This is just an example on how to change the wall_heights online.
    # For your project, you will likely want to track training progress in the environment
    # And set the wall positions based on that (increasing difficulty)
    # For instance, you could pass another parameter to the environment init that specifies training length
    # And then count the number of resets, or set it externally, and set wall heights based on that.
    
    # OLD SAMPLING FUNCTION
    #   def sample_wall_heights(self, rng, range_min=0.2, range_max=0.2):
    #     """Sample random wall heights (JAX-compatible)."""
    #     rng, height_key = jax.random.split(rng)
    #     num_walls = len(self._wall_geom_ids)
    #     wall_heights = jax.random.uniform(
    #         height_key,
    #         shape=(num_walls,),
    #         minval=range_min,
    #         maxval=range_max
    #     )
    #     return wall_heights, rng

    # Deterministic curriculum: all wall heights increase linearly with progress
    # def sample_wall_heights_deterministic(self, range_min=0.2, range_max=0.4):
    #     """Deterministically set all wall heights to the current curriculum value (no randomness), or fixed for evaluation."""
    #     num_walls = len(self._wall_geom_ids)
    #     if getattr(self, '_curriculum_learning', False):
    #         wall_heights = jp.ones((num_walls,)) * range_max
    #     else:
    #         current_step = self._reset_count * self._config.episode_length
    #         progress = min(current_step / self._total_steps, 1.0)
    #         wall_height = range_min + (range_max - range_min) * progress
    #         wall_heights = jp.ones((num_walls,)) * wall_height
    #     return wall_heights

    # Curriculum learning: wall heights sampled every reset, up to range_max at total_steps
    def sample_wall_heights(self, rng, range_min=0.03, range_max=0.06, reset_count: Optional[jax.Array] = None):
        num_walls = len(self._wall_geom_ids)
        if not getattr(self, '_curriculum_learning', False):
            # Use fixed wall height
            wall_heights = jp.ones((num_walls,)) * range_max
            return wall_heights, rng
        else:
            # Curriculum: sample upper value between range_min and range_max
            total_steps = self._total_steps
            total_steps = jp.maximum(total_steps, 1.0)
            current_step = reset_count * self._episode_length # Not accurate, just an estimate if the episode runs to the end
            progress = jp.clip(current_step / self._total_steps, 0.0, 1.0)
            max_wall_height = range_min + (range_max - range_min) * progress
            rng, height_key = jax.random.split(rng)
            wall_heights = jax.random.uniform(
                height_key,
                shape=(num_walls,),
                minval=range_min,
                maxval=max_wall_height
            )
            return wall_heights, rng


    def set_wall_mocap_positions(self, data, wall_heights):
        """Set wall positions using mocap control."""
        # Get original wall body positions
        new_mocap_pos = data.mocap_pos
        for i, (body_id, height) in enumerate(zip(self._wall_body_ids, wall_heights)):
            mocap_id = body_id - self.mjx_model.nbody + self.mjx_model.nmocap  # Convert to mocap index

            # Get current position and update Z coordinate
            current_pos = new_mocap_pos[mocap_id]
            new_pos = current_pos.at[2].set(height)
            new_mocap_pos = new_mocap_pos.at[mocap_id].set(new_pos)

        return data.replace(mocap_pos=new_mocap_pos)
    

    def _increment_reset_counter(self) -> jax.Array:
        def _inc_host(_):
            # Increment host-side counter and return it as a NumPy scalar/array
            self._reset_count += 1
            import numpy as _np
            return _np.int32(self._reset_count)

        # io_callback signature: io_callback(callback, result_shape_dtypes, *args)
        return io_callback(
            _inc_host,
            jax.ShapeDtypeStruct((), jp.int32),
            jp.array(0, dtype=jp.int32),
        )
    # ==========================================================================

    def reset(self, rng: jax.Array) -> mjx_env.State:

        qpos = self._init_q # Start from default "home" joint positions
        qvel = jp.zeros(self.mjx_model.nv) # Start from zero velocity

        # x=+U(-0.1, 0.1), y=+U(-0.1, 0.1), yaw=U(-pi, pi).
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.1, maxval=0.1)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)

        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)

        # Sample wall heights for this episode using a runtime (JIT-visible) counter
        rng, wall_rng = jax.random.split(rng)
        reset_count = self._increment_reset_counter()

        jax.debug.print("DEBUG reset: _reset_count= {}, curriculum_learning= {}", reset_count, self._curriculum_learning)

        wall_heights, _ = self.sample_wall_heights(wall_rng, reset_count=reset_count)

        # d(xyzrpy)=U(-0.5, 0.5)
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(
            jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        )

        # Create data with original model (since wall heights are part of qpos)
        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=qpos[7:19],  # Only robot leg joints, not wall joints
            impl=self.mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        # Run forward pass with the original model
        data = mjx.forward(self.mjx_model, data)

        # Set wall heights using mocap control
        data = self.set_wall_mocap_positions(data, wall_heights) # CHECK

        rng, key1, key2, key3 = jax.random.split(rng, 4)
        time_until_next_pert = jax.random.uniform(
            key1,
            minval=self._config.pert_config.kick_wait_times[0],
            maxval=self._config.pert_config.kick_wait_times[1],
        )
        steps_until_next_pert = jp.round(time_until_next_pert / self.dt).astype(
            jp.int32
        )
        pert_duration_seconds = jax.random.uniform(
            key2,
            minval=self._config.pert_config.kick_durations[0],
            maxval=self._config.pert_config.kick_durations[1],
        )
        pert_duration_steps = jp.round(pert_duration_seconds / self.dt).astype(
            jp.int32
        )
        pert_mag = jax.random.uniform(
            key3,
            minval=self._config.pert_config.velocity_kick[0],
            maxval=self._config.pert_config.velocity_kick[1],
        )

        rng, key1, key2 = jax.random.split(rng, 3)
        time_until_next_cmd = jax.random.exponential(key1) * 5.0
        steps_until_next_cmd = jp.round(time_until_next_cmd / self.dt).astype(
            jp.int32
        )
        # Use our single-velocity sampling logic
        initial_cmd = jp.zeros(3)  # Start with zero command
        cmd = self.sample_command(key2, initial_cmd)
        w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
        initial_yaw = jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        # Convert local command to world coordinates
        cos_yaw = jp.cos(initial_yaw)
        sin_yaw = jp.sin(initial_yaw)
        world_cmd = jp.array([
            cos_yaw * cmd[0] - sin_yaw * cmd[1],
            sin_yaw * cmd[0] + cos_yaw * cmd[1],
            cmd[2]
        ])
        info = {
            "rng": rng,
            "command": cmd,
            "world_command": world_cmd,  # NEW: Initially same as local command
            "steps_until_next_cmd": steps_until_next_cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),
            "steps_until_next_pert": steps_until_next_pert,
            "pert_duration_seconds": pert_duration_seconds,
            "pert_duration": pert_duration_steps,
            "steps_since_last_pert": 0,
            "pert_steps": 0,
            "pert_dir": jp.zeros(3),
            "pert_mag": pert_mag,
            "last_knee_contact": jp.zeros(4, dtype=bool), # Add knee contact, 4 because of 4 knees
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)


    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

        #jax.debug.print("DEBUG step: _reset_count= {}, curriculum_learning= {}", self._reset_count, self._curriculum_learning)

        # Apply perturbation if enabled
        if self._config.pert_config.enable:
            state = self._maybe_apply_perturbation(state)

        # Compute target joints for the robot
        motor_targets = self._default_pose + action * self._config.action_scale

        # Advance the simulation using the current state and the computed motor targets
        # Wall heights are now part of qpos, so we use the original model
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )

        # CHECK
        # NOTE: You can do something similar to check for knee collision if you want to integrate this in your reward.
        # Vectorized feet-floor collision detection
        contact = jax.vmap(lambda geom_id: collision.geoms_colliding(data, geom_id, self._floor_geom_id))(
            self._feet_geom_id
        ) # Boolean array: one entry per foot, True if in contact with the floor
        # Vectorized feet-wall collision detection
        feet_wall_collisions = jax.vmap(
            lambda foot_geom_id: jax.vmap(
                lambda wall_geom_id: collision.geoms_colliding(data, foot_geom_id, wall_geom_id)
            )(self._wall_geom_ids)
        )(self._feet_geom_id) # 2D boolean array: shape (num_feet, num_walls), True if a foot is in contact with a wall
        # Check if any foot collides with any wall
        contact = contact | feet_wall_collisions.any(axis=1) # Updates the contact array: a foot is considered in contact if it touches the floor or any wall

        contact_filt = contact | state.info["last_contact"] # Combines current and previous contacts to avoid missing brief contacts

        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt # True for feet that have just made contact after being in the air
        state.info["feet_air_time"] += self.dt # Increment air time for all feet (later reset for feet that are in contact)
        
        # -----------------------------
        # NOTE: Implementation of knee rewards below:
        knee_wall_collisions = jax.vmap(
            lambda knee_geom_id: jax.vmap(
                lambda wall_geom_id: collision.geoms_colliding(data, knee_geom_id, wall_geom_id)
            )(self._wall_geom_ids)
        )(self._knee_geom_id) # 2D boolean array: shape (num_knees, num_walls), True if a knee is in contact with a wall
        
        knee_contact = knee_wall_collisions.any(axis=1) # jnp.Array of shape (num_knees,)
        # -----------------------------

        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]  # Absolute foot height
        terrain_distance = self._get_terrain_height_below_feet(self._mj_model, data)
        terrain_height = p_fz - terrain_distance  # Terrain z-coordinate below foot
        relative_swing_height = p_fz - terrain_height  # Height above terrain
        state.info["swing_peak"] = jp.where(
            contact,
            state.info["swing_peak"],  # Don't update if in contact
            jp.maximum(state.info["swing_peak"], relative_swing_height)
        ) # Updates the peak swing height for each foot, but only if the foot is not in contact (i.e., during swing phase)

        obs = self._get_obs(data, state.info)
        done = self._get_termination(data) # Check if the episode should terminate (e.g., if the robot has fallen)

        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact, knee_contact
        )
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        } # Scales each reward by its configured weight
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0) # Total reward, clipped to avoid extreme values

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["steps_until_next_cmd"] -= 1
        
        # NEW: Handle world_command updates when new commands are issued
        state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
        new_local_cmd = self.sample_command(key1, state.info["command"])
        
        # Get current robot yaw to convert new local command to world coordinates
        trunk_quat = data.qpos[3:7]
        w, x, y, z = trunk_quat[0], trunk_quat[1], trunk_quat[2], trunk_quat[3]
        current_yaw = jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        # Convert new local command to world coordinates
        cos_yaw = jp.cos(current_yaw)
        sin_yaw = jp.sin(current_yaw)
        new_world_cmd = jp.array([
            cos_yaw * new_local_cmd[0] - sin_yaw * new_local_cmd[1],
            sin_yaw * new_local_cmd[0] + cos_yaw * new_local_cmd[1],
            new_local_cmd[2]
        ])
        
        # Update both local and world commands when new command is issued
        state.info["command"] = jp.where(
            state.info["steps_until_next_cmd"] <= 0,
            new_local_cmd,
            state.info["command"],
        )
        
        # NEW: Update world command only when new command is issued
        state.info["world_command"] = jp.where(
            state.info["steps_until_next_cmd"] <= 0,
            new_world_cmd,
            state.info["world_command"]
        )
        
        state.info["steps_until_next_cmd"] = jp.where(
            done | (state.info["steps_until_next_cmd"] <= 0),
            jp.round(jax.random.exponential(key2) * 5.0 / self.dt).astype(jp.int32),
            state.info["steps_until_next_cmd"],
        )
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        # NOTE: Update last_knee_contact
        state.info["last_knee_contact"] = knee_contact
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state


    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_upvector(data)[-1] < 0.0
        return fall_termination


    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> Dict[str, jax.Array]:
        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        ) # Add uniform noise to gyro readings

        gravity = self.get_gravity(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        ) # Add uniform noise to gravity vector

        # Only include robot joint angles, not wall joints
        # Robot joints are at indices [7:19] (12 leg joints)
        # Wall joints come after robot joints, so robot is still [7:19]
        joint_angles = data.qpos[7:19]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_pos
        ) # Add uniform noise to joint angles

        joint_vel = data.qvel[6:18]  # Only robot joint velocities
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        ) # Add uniform noise to joint velocities

        linvel = self.get_local_linvel(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_linvel = (
            linvel
            + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.linvel
        ) # Add uniform noise to linear velocity

        state = jp.hstack([
            noisy_linvel,  # 3, range [ ]
            noisy_gyro,  # 3, range [ ] 
            noisy_gravity,  # 3, range [ ] check if gravity or acc
            noisy_joint_angles - self._default_pose,  # 12. 
            noisy_joint_vel,  # 12. 
            info["last_act"],  # 12 
            info["command"],  # 3
        ])

        accelerometer = self.get_accelerometer(data)
        angvel = self.get_global_angvel(data)

        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()

        privileged_state = jp.hstack([
            info["last_act"],  # 12
            info["command"],  # 3
            gyro,  # 3
            accelerometer,  # 3
            gravity,  # 3
            linvel,  # 3
            angvel,  # 3
            joint_angles - self._default_pose,  # 12 (offset from default pose)
            joint_vel,  # 12
            data.actuator_force,  # 12
            info["last_contact"],  # 4
            feet_vel,  # 4*3
            info["feet_air_time"],  # 4
            data.xfrc_applied[self._torso_body_id, :3],  # 3 (force applied to torso)
            info["steps_since_last_pert"] >= info["steps_until_next_pert"],  # 1 (bool, 1 if perturbation is active),
            info["last_knee_contact"],      # Add this for future teacher-student implementation
        ])


         # ------------- NEW for exteroceptive -------------
        # If exteroceptive flag is set, append exteroceptive data
        if self._exteroceptive:
            terrain_height = self._get_exteroceptive(data)["terrain_height"]
            # Add noise to terrain height
            info["rng"], noise_rng = jax.random.split(info["rng"])
            noisy_terrain_height = (
                terrain_height
                + (2 * jax.random.uniform(noise_rng, shape=terrain_height.shape) - 1)
                * self._config.noise_config.level
                * self._config.noise_config.scales.extero
            )
            state = jp.hstack([state, noisy_terrain_height])
            privileged_state = jp.hstack([privileged_state, terrain_height])
        # ---------------------------------------------------

        return {
            "state": state,
            "privileged_state": privileged_state,
            "student_state": state,  # CHECK For potential future use
        }


    def _get_exteroceptive(self, data: mjx.Data) -> Dict:
        """Get terrain height in front of and to the sides of the robot with multiple rays for robustness."""

        # Extracts the torso's queaternion data
        quaternion_data = data.xquat[self._torso_body_id]
        w = quaternion_data[0] # rotation
        x = quaternion_data[1] # x coord
        y = quaternion_data[2] # y coord
        z = quaternion_data[3] # z coord

        # Convert to heading angle in the XY plane (rotation around z-axis) from its quaternion orientation
        yaw = jp.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        # Define grid of virtual sensors
        offsets = jp.array([
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, -1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, -1.0],
            [-1.0, 1.0],
            [-1.0, 0.0],
            [-1.0, -1.0]
        ])

        front_dict = self._get_grid_mean(offsets, yaw, x_shift=0.5, y_shift=0, x_scale_factor=0.07, y_scale_factor=0.3, data=data)
        left_dict = self._get_grid_mean(offsets, yaw, x_shift=-0.2, y_shift=0.3, x_scale_factor=0.15, y_scale_factor=0.07, data=data)
        right_dict = self._get_grid_mean(offsets, yaw, x_shift=-0.2, y_shift=-0.3, x_scale_factor=0.15, y_scale_factor=0.07, data=data)

        dicts = [front_dict, left_dict, right_dict]

        grids_data = {}

        grids_data["distances"] = jp.concatenate([d["distances"] for d in dicts], axis=0)

        grids_data["origins"] = jp.vstack([d["origins"] for d in dicts])

        grids_data["directions"] = dicts[0]["directions"]
        
        mean_height_array = jp.array([
            front_dict["terrain_height"],
            left_dict['terrain_height'],
            right_dict['terrain_height'],
        ])

        return {
            "terrain_height":  mean_height_array,
            "distances": grids_data["distances"],
            "directions": grids_data["directions"],
            "origins": grids_data["origins"],
        }


    def _get_grid_mean(self, offsets, yaw, x_shift, y_shift, x_scale_factor, y_scale_factor, data):

        torso_pos = data.xpos[self._torso_body_id]

        # Get 2D yaw rotation (apply to the XY plane)
        c = jp.cos(yaw)
        s = jp.sin(yaw)

        # Ray direction: straight down in world coordinates
        ray_dir = jp.array([0.0, 0.0, -1.0])

        # Scale and position grid in relation ro robot
        z_shift = 0.3

        # Apply scale and forward shift in the torso frame
        local_x = offsets[:, 0] * x_scale_factor + x_shift
        local_y = offsets[:, 1] * y_scale_factor + y_shift
        local_z = z_shift

        # Calculate rotated coordinates in world frame
        world_x = c * local_x - s * local_y
        world_y = s * local_x + c * local_y
        world_z = torso_pos[2] + local_z

        # Combine into world ray starts: torso_pos + rotated local XY, and z = torso_z + z_shift
        ray_starts = jp.stack([
            torso_pos[0] + world_x,
            torso_pos[1] + world_y,
            jp.ones_like(world_x) * world_z
        ], axis=1)

        distances, _ = ray.batch_ray(
            self._mj_model, data, ray_starts, ray_dir, (),
            True, bodyexclude=self._robot_body_ids
        )

        mean_distance = distances
        terrain_height = mean_distance - z_shift

        return {
            "terrain_height": terrain_height,
            "distances": distances,
            "directions": ray_dir,
            "origins": ray_starts,
        }


    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
        knee_contact: jax.Array,      # Add knee_contact into reward
    ) -> dict[str, jax.Array]:
        del metrics  # Unused, delete to prevent "unused variable" warning
        return {
            "tracking_lin_vel": self._reward_tracking_lin_vel(
                info["command"], self.get_local_linvel(data)
            ),
            "tracking_ang_vel": self._reward_tracking_ang_vel(
                info["command"], self.get_gyro(data)
            ),
            "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
            "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
            "orientation": self._cost_orientation(self.get_upvector(data)),
            "stand_still": self._cost_stand_still(info["command"], data.qpos[7:19]),
            "termination": self._cost_termination(done),
            "pose": self._reward_pose(data.qpos[7:19]),
            "torques": self._cost_torques(data.actuator_force),
            "action_rate": self._cost_action_rate(
                action, info["last_act"], info["last_last_act"]
            ),
            "energy": self._cost_energy(data.qvel[6:18], data.actuator_force),
            "feet_slip": self._cost_feet_slip(data, contact, info),
            "feet_clearance": self._cost_feet_clearance(data, contact),
            "feet_air_time": self._reward_feet_air_time(
                info["feet_air_time"], first_contact, info["command"]
            ),
            "torso_height": self._cost_torso_height(data),
            "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:19]),
            "knee_collisions": self._cost_knee_collisions(data, knee_contact)   # Add knee collision cost
        }

    # --------------
    # NOTE: Implementation of knee collision cost
    def _cost_knee_collisions(self, data: mjx.Data, knee_contact: jax.Array) -> jax.Array:
        """Penalize knees colliding with walls"""
        knees_in_contact = jp.sum(knee_contact.astype(jp.float32))   # Counts amount of "True" values
        cost = knees_in_contact/len(knee_contact)
        return cost
    # --------------

    def _cost_torso_height(self, data: mjx.Data) -> jax.Array:
        """Penalize deviation from target torso height above terrain."""
        height_above_terrain = self._get_torso_terrain_height(data)
        target_height = self._config.reward_config.desired_torso_height
        
        # Squared error from target height
        return jp.square(height_above_terrain - target_height)
  

    # Tracking rewards.
    def _reward_tracking_lin_vel(
        self,
        commands: jax.Array,
        local_vel: jax.Array,
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes).
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        return jp.exp(-lin_vel_error)

    def _reward_tracking_ang_vel(
        self,
        commands: jax.Array,
        ang_vel: jax.Array,
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw).
        ang_vel_error = jp.square(commands[2] - ang_vel[2])
        return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)


    # Base-related rewards.
    def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
        # Penalize z axis base linear velocity.
        return jp.square(global_linvel[2])

    def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
        # Penalize xy axes base angular velocity.
        return jp.sum(jp.square(global_angvel[:2]))

    def _cost_orientation(self, torso_zaxis: jax.Array,
                            tolerance: float = 0.15,   ## Consider increasing this a bit
                            exp_scale: float = 7.0) -> jax.Array:
        """
        Penalize orientation outside a tolerance range with exponential growth.

        Args:
            torso_zaxis: Z-axis vector of torso orientation
            tolerance: Half-width of the dead zone (no penalty range) in radians
            exp_scale: Scale factor for exponential penalty growth

        Returns:
            Cost value (0 within tolerance, exponentially growing outside)
        """
        # Get pitch and roll components (first two elements)
        pitch_roll = torso_zaxis[:2]

        # Calculate absolute deviations
        abs_deviations = jp.abs(pitch_roll)

        # Calculate excess beyond tolerance (clipped to 0 if within tolerance)
        excess = jp.maximum(0.0, abs_deviations - tolerance)

        # Apply exponential penalty to excess
        penalties = jp.expm1(exp_scale * excess)  # expm1(x) = exp(x) - 1

        # Sum penalties for both pitch and roll
        return jp.sum(penalties)


    # Energy related rewards.
    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques.
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _cost_energy(
        self, qvel: jax.Array, qfrc_actuator: jax.Array
    ) -> jax.Array:
        # Penalize energy consumption.
        return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

    def _cost_action_rate(
        self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ) -> jax.Array:
        del last_last_act  # Unused.
        return jp.sum(jp.square(act - last_act))


    # Other rewards.
    def _reward_pose(self, qpos: jax.Array) -> jax.Array:
        # Stay close to the default pose.
        weight = jp.array([1.0, 1.0, 0.1] * 4)
        return jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * weight))

    def _cost_stand_still(
        self,
        commands: jax.Array,
        qpos: jax.Array,
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(commands)
        return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.01)

    def _cost_termination(self, done: jax.Array) -> jax.Array:
        # Penalize early termination.
        return done

    def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
        # Penalize joints if they cross soft limits.
        out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0) # Clips negative values to 0. Negative makes the cost positive when a joint is too low
        out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None) # Clips positive values to 0
        return jp.sum(out_of_limits)


    # Feet related rewards.
    def _reward_feet_air_time(
        self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        # Reward air time.
        cmd_norm = jp.linalg.norm(commands)
        rew_air_time = jp.sum(jp.exp(-jp.square(air_time - self._config.reward_config.desired_foot_air_time)) * first_contact)
        rew_air_time *= cmd_norm > 0.01  # No reward for close to zero commands.
        return rew_air_time
  
    def _cost_feet_slip(
        self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(info["command"])
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
        return jp.sum(vel_xy_norm_sq * contact) * (cmd_norm > 0.01)


    def _get_torso_terrain_height(self, data: mjx.Data) -> jax.Array:
        """Get torso height above terrain using multiple rays for robustness."""
        torso_pos = data.xpos[self._torso_body_id] # 3D position of torso in world coordinates
        
        # Cast rays in a small pattern around torso center
        offsets = jp.array([
            [0.0, 0.0],      # Center
            [0.1, 0.0],      # Forward
            [-0.1, 0.0],     # Back
            [0.0, 0.1],      # Right
            [0.0, -0.1],     # Left
        ])
        
        # Create ray start positions
        ray_starts = []
        for offset in offsets:
            start_pos = torso_pos + jp.array([offset[0], offset[1], 0.5])
            ray_starts.append(start_pos)
        
        ray_starts = jp.array(ray_starts)
        ray_dir = jp.array([0.0, 0.0, -1.0]) # All rays point straight down (negative Z direction)
        
        # Batch ray cast
        distances, _ = ray.batch_ray(
            self._mj_model, data, ray_starts, ray_dir, (),
            True, bodyexclude=self._robot_body_ids
        ) # Returns the distance from each start point to the first terrain hit below
        
        # Use minimum distance (highest terrain point under torso)
        min_distance = jp.min(distances)
        height_above_terrain = min_distance - 0.5
        
        return height_above_terrain


    def _get_terrain_height_below_feet(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
        """Sample terrain height around each foot with a single batch ray cast."""
        foot_pos = data.site_xpos[self._feet_site_id]  # Shape: (4, 3) 3D world positions of the 4 feet
        
        # Define sampling pattern (e.g., 5 points per foot: center + 4 around)
        # These are XY offsets from each foot
        sample_offsets = jp.array([
        # Center point
        [0.0, 0.0],

        # Close cardinal directions (0.05 units)
        [0.05, 0.0],      # Forward
        [-0.05, 0.0],     # Back  
        [0.0, 0.05],      # Right
        [0.0, -0.05],     # Left

        # Close diagonal directions (0.05 units)
        [0.05, 0.05],     # Forward-right
        [0.05, -0.05],    # Forward-left
        [-0.05, 0.05],    # Back-right
        [-0.05, -0.05],   # Back-left

        # # # # Far cardinal directions (0.1 units)
        # [0.2, 0.0],       # Far forward
        # [-0.2, 0.0],      # Far back
        # [0.0, 0.2],       # Far right
        # [0.0, -0.2],      # Far left

        # # # Far diagonal directions (0.2 units)
        # [0.2, 0.2],       # Far forward-right
        # [0.2, -0.2],      # Far forward-left
        # [-0.2, 0.2],      # Far back-right
        # [-0.2, -0.2],     # Far back-left
        ])  # Shape: (17, 2)
      
        # Create all sample positions for all feet at once
        # Broadcast foot positions with offsets
        num_samples = sample_offsets.shape[0]
        
        # Expand dimensions for broadcasting
        foot_pos_expanded = foot_pos[:, None, :]  # (4, 1, 3)
        offsets_3d = jp.pad(sample_offsets[None, :, :], ((0,0), (0,0), (0,1)))  # (1, 5, 3)
        # print(foot_pos)
        # All sample positions: (4 feet * 5 samples = 20 total)
        sample_positions = (foot_pos_expanded + offsets_3d).reshape(-1, 3)  # (20, 3)
        # print(sample_positions)

        # Start rays 0.5 units above the sample positions
        ray_start_offset = jp.array([0.0, 0.0, 0.5])
        elevated_sample_positions = sample_positions + ray_start_offset  # (20, 3)
        
        # Single ray direction for all samples (straight down)
        ray_dir = jp.array([0.0, 0.0, -1.0])
        
        # Single batch ray cast for all samples
        distances, _ = ray.batch_ray(
            model, data, elevated_sample_positions, ray_dir, (),
            True, bodyexclude=self._robot_body_ids
        )  # Shape: (20,)
        # print(geom_ids.min())
        # print(self._robot_body_ids)
        # Reshape to (4 feet, 5 samples) and take max height around each foot
        distances_per_foot = distances.reshape(4, num_samples)
        
        # Get the minimum distance (highest terrain) around each foot
        # Note: smaller distance = higher terrain when casting downward
        min_distances = jp.min(distances_per_foot, axis=1)  # Shape: (4,)
        
        # Subtract the 0.5 offset to get terrain height relative to original foot position
        # Positive values = terrain below foot, Negative values = terrain above foot
        terrain_heights = min_distances - 0.5
        
        return terrain_heights


    def _cost_feet_clearance(self, data: mjx.Data, contact: jax.Array) -> jax.Array:
        """Penalize insufficient clearance during swing phase only."""
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr] # 3D velocities of each foot (4, 3)
        vel_xy = feet_vel[..., :2] # XY components of foot velocities
        vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1)) # Norm of XY velocities (4,)
        
        # Get terrain clearance
        clearance = self._get_terrain_height_below_feet(self._mj_model, data)
        
        # Minimum safe clearance
        min_safe_clearance = self._config.reward_config.max_foot_height
        
        # Only penalize insufficient clearance, and only during swing phase
        insufficient_clearance = jp.maximum(0, min_safe_clearance - clearance)

        # ~contact is True during swing phase. vel_norm scales penalty by foot speed (less penalty for slow movement)
        return jp.sum(insufficient_clearance * vel_norm * (~contact)) 


    # Perturbation and command sampling.
    def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
        # Generate a random horizontal direction vector 
        def gen_dir(rng: jax.Array) -> jax.Array:
            angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
            return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

        # 
        def apply_pert(state: mjx_env.State) -> mjx_env.State:
            t = state.info["pert_steps"] * self.dt # Current time into the perturbation
            u_t = 0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"]) # Smooth perturbation profile
            # kg * m/s * 1/s = m/s^2 = kg * m/s^2 (N).
            force = (
                u_t  # (unitless)
                * self._torso_mass  # kg
                * state.info["pert_mag"]  # m/s
                / state.info["pert_duration_seconds"]  # 1/s
            ) # Newtons: F = m * v / t
            xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
            xfrc_applied = xfrc_applied.at[self._torso_body_id, :3].set(
                force * state.info["pert_dir"]
            )
            data = state.data.replace(xfrc_applied=xfrc_applied)
            state = state.replace(data=data)
            state.info["steps_since_last_pert"] = jp.where(
                state.info["pert_steps"] >= state.info["pert_duration"],
                0,
                state.info["steps_since_last_pert"],
            )
            state.info["pert_steps"] += 1
            return state

        def wait(state: mjx_env.State) -> mjx_env.State:
            state.info["rng"], rng = jax.random.split(state.info["rng"])
            state.info["steps_since_last_pert"] += 1
            xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
            data = state.data.replace(xfrc_applied=xfrc_applied)
            state.info["pert_steps"] = jp.where(
                state.info["steps_since_last_pert"]
                >= state.info["steps_until_next_pert"],
                0,
                state.info["pert_steps"],
            )
            state.info["pert_dir"] = jp.where(
                state.info["steps_since_last_pert"]
                >= state.info["steps_until_next_pert"],
                gen_dir(rng),
                state.info["pert_dir"],
            )
            return state.replace(data=data)

        return jax.lax.cond(
            state.info["steps_since_last_pert"]
            >= state.info["steps_until_next_pert"],
            apply_pert,
            wait,
            state,
        )


    def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
        rng, choice_rng, y_rng, w_rng, z_rng = jax.random.split(rng, 5)
        
        # Choose which type of velocity to sample (0: forward/back, 1: left/right, 2: angular)
        velocity_type = jax.random.choice(choice_rng, 3)
        
        # Sample values for all dimensions
        y_k = jax.random.uniform(
            y_rng, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
        )
        z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(3,)) # Decides (with prob cmd_b) whether to use the new sampled value or keep the old one
        w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,)) # For each dimension, randomly decides (50% chance) whether to update the value or keep the old one
        
        # Create mask to only update one velocity type
        velocity_mask = jp.zeros(3)
        velocity_mask = velocity_mask.at[velocity_type].set(1.0)
        
        # Apply mask so only one velocity type gets updated, others stay at current value
        x_kp1 = x_k - w_k * velocity_mask * (x_k - y_k * z_k)
        return x_kp1