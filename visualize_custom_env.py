#!/usr/bin/env python3
"""Visualization script for Go1 with height scanner."""
import os
# Tell XLA to use Triton GEMM for faster GPU matrix multiplication in JAX
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags
os.environ['MUJOCO_GL'] = 'egl' # Use EGL for offscreen rendering with MuJoCo

import jax
import jax.numpy as jp
import numpy as np
import mujoco
import cv2
from PIL import Image, ImageDraw, ImageFont
from custom_env import Joystick, default_config

def main():
    # Set up visualization options (later used in env.render())
    scene_option = mujoco.MjvOption() # Holds visualization options settings for the MuJoCo scene
    scene_option.geomgroup[2] = True   # Show visual geoms (e.g. robot body, walls, floor)
    scene_option.geomgroup[3] = False  # Hide collision geoms
    scene_option.geomgroup[5] = True   # Show sites (including height scanner visualization)
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True  # Show contact points
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True
    print("Creating Visualization...")

    # Perform multiple random resets to show different wall configurations
    num_resets = 5
    steps_per_reset = 40  # More steps to show movement

    # We will no longer use the registry, but directly load our custom XML and model.
    # We will load the "debug" version here, which adds an additional wall *under* the robot
    # We add this to understand that changing walls correctly affects collision + raycasting.
    xml_path = 'custom_env_debug_wall.xml'
    env = Joystick(xml_path=xml_path, config=default_config(), total_steps=num_resets*steps_per_reset, steps_per_reset=steps_per_reset)

    # ----------------------------
    # NOTE: Added functionality to visualize ray casting
    # Get body ids for the mocap bodies
    ray_count = 9
    origin_body_ids = [env._mj_model.body(f"ray_mocap_origin_{i}").id for i in range(ray_count)]
    endpoint_body_ids = [env._mj_model.body(f"ray_mocap_endpoint_{i}").id for i in range(ray_count)]

    # Retireve nr of bodies for ray viz
    mjx_model = env._mjx_model
    nbody = mjx_model.nbody
    nmocap = mjx_model.nmocap

    # Get model and site IDs for ray visualization
    model = env._mj_model
    origin_site_ids = [model.site(f"ray_origin_{i}").id for i in range(9)]
    endpoint_site_ids = [model.site(f"ray_endpoint_{i}").id for i in range(9)]
    # ----------------------------

    # NOTE: For this test, we manually set init_q z position high to avoid collisions with walls
    env._init_q = env._init_q.at[2].set(1.0)

    # JIT compile the functions for speed
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_terrain_height = jax.jit(env._get_torso_terrain_height)
    jit_scanned_terrain_height = jax.jit(env._get_exteroceptive)    # Get exteroceptive data 

    # Initialize
    key = jax.random.PRNGKey(15)

    os.makedirs("gifs", exist_ok=True) # Ensure output directory exists

    print("Running simulation...")
    for reset_idx in range(num_resets):
        print(f"Creating GIF {reset_idx+1}/{num_resets}")

        # Randomize wall heights for this reset
        #set_wall_heights(env, range_min=0.05, range_max=0.4)

        # Generate new random key for this reset
        key, reset_key = jax.random.split(key)

        # Reset to new random position
        state = jit_reset(reset_key)

        # Collect frames and terrain heights for this reset
        rollout = []
        terrain_heights = []
        scanned_terrain_heights = []    # Add list to save exteroceptive data

        for step in range(steps_per_reset):
            rollout.append(state)

            # Calculate terrain height for this state
            terrain_height = jit_terrain_height(state.data)
            terrain_heights.append(float(terrain_height))

            # -------------------------------------
            # NOTE: Section for visualizing terrain height/rays
            # Calculate scanned terrain height (our values)
            scanned_terrain_height = jit_scanned_terrain_height(state.data)
            scanned_terrain_heights.append(float(scanned_terrain_height["terrain_height"])) # Why is this done? Scrutinize!

            # Ray visualization logic
            # Get ray data (assumed to return dict with keys 'origins', 'directions', 'distances')
            ray_data = scanned_terrain_height
            origins = np.array(ray_data["origins"])  # shape (9, 3)
            directions = np.array(ray_data["directions"])  # shape (9, 3)
            distances = np.array(ray_data["distances"])  # shape (9,)
            endpoints = origins + directions * distances[:, None]

            mocap_pos = state.data.mocap_pos  # jax array

            # Set origins
            for i, body_id in enumerate(origin_body_ids):
                mocap_index = int(body_id - nbody + nmocap)
                origin_jax = jp.asarray(origins[i])
                mocap_pos = mocap_pos.at[mocap_index].set(origin_jax)

            # Set endpoints
            for i, body_id in enumerate(endpoint_body_ids):
                mocap_index = int(body_id - nbody + nmocap)
                endpoint_jax = jp.asarray(endpoints[i])
                mocap_pos = mocap_pos.at[mocap_index].set(endpoint_jax)

            # write back into state
            state = state.replace(data=state.data.replace(mocap_pos=mocap_pos))
            # -------------------------------------

            # Check for NaN/inf in reward
            if not jp.isfinite(state.reward):
                print(f"    WARNING: Non-finite reward detected!")
                print(f"    Reward value: {state.reward}")
                raise ValueError("Non-finite reward encountered during simulation")

            # Use small random actions to show some movement
            action_key, key = jax.random.split(key)
            action = jax.random.normal(action_key, (env.action_size,)) * 0.1
            state = jit_step(state, action)

        # Debug prints: check wall positions from simulation data
        current_data = rollout[-1].data  # Get the latest state
        print(f"\nReset {reset_idx+1} - Wall positions:")
        for i, wall_geom_id in enumerate(env._wall_geom_ids):
            wall_pos = current_data.geom_xpos[wall_geom_id]
            print(f"  Wall {i}: geom_xpos[z]={wall_pos[2]:.3f}")
        print()

        # Render frames for this reset
        print(f"    Rendering {len(rollout)} frames...")

        frames = env.render(
            rollout,
            camera="track",  # Use tracking camera
            scene_option=scene_option,
            width=480,
            height=360,
        )

        # Create GIF for this reset
        #gif_filename = f'wall_randomization_{reset_idx+1:02d}7.gif'
        gif_filename = os.path.join("gifs", f"wall_randomization_{reset_idx+1:02d}7.gif")
        print(f"    Saving GIF to {gif_filename}...")

        # Convert frames to PIL Images and add terrain height overlay
        pil_images = []
        for i, frame in enumerate(frames):
            pil_image = Image.fromarray(frame) # Convert NumPy array to PIL Image

            # Add terrain height overlay
            draw = ImageDraw.Draw(pil_image)
            terrain_height_text = f"Scanned Terrain Height: {scanned_terrain_heights[i]:.3f}m"

            # Try to use a default font, fallback to default if not available
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()

            # Draw background rectangle for better text visibility
            text_bbox = draw.textbbox((0, 0), terrain_height_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Position text in top-left corner with padding
            x, y = 10, 10
            draw.rectangle([x-5, y-5, x+text_width+5, y+text_height+5], fill=(0, 0, 0, 128))
            draw.text((x, y), terrain_height_text, fill=(255, 255, 255), font=font)

            pil_images.append(pil_image)

        # Save as GIF with appropriate duration
        pil_images[0].save(
            gif_filename,
            save_all=True,
            append_images=pil_images[1:],
            duration=50,  # 50ms per frame = 20 fps
            loop=0
        )

        print(f"    ✓ GIF saved successfully to {gif_filename}")

    print(f"✓ All {num_resets} GIFs created successfully!")
    

if __name__ == "__main__":
    print("JAX devices:", jax.devices())
    main()