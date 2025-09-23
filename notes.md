# QUESTIONS
- I think the code already uses raycasting to measure torso height. Which other directions should we measure? Maybe forward and sideways, all angled slightly downward? A ring around the torso downward?


- Check how to integrate the new measurements into the reward function.


# NOTES
- Set steps_per_reset to maximum 40 to reduce GPU memory usage and avoid RuntimeError: INTERNAL: cuSolver internal error.
- Search for 'CHECK' to find places to modify/add code.


# USEFUL COMMANDS:

- `deactivate` to exit the virtual environment. 
- `watch -n 1 nvidia-smi` to monitor GPU usage in real-time (run in a separate terminal).




---
# In-depth Explanation: Codebase Changes & Adding Exteroceptive Information

## 1. Environment Definition (`custom_env.py`)
- Main file for defining/modifying the simulation environment.
- Add exteroceptive sensors (e.g., distance) here. Implement logic for the agent to sense walls, floor height, etc.
- Modify the reward function to use new sensor data.

## 2. Task Specification (`custom_env.xml`, `custom_env_debug_wall.xml`)
- XML files define the physical layout (robot, walls, obstacles).
- The only functional difference is the presence of the additional debug wall in custom_env_debug_wall.xml, which is used for development and troubleshooting.

## 3. Knee Collisions
- Simulation detects knee collisions; knee IDs are available.
- Penalize knee collisions in the reward function if desired.

## 4. Environment Randomization / Curriculum Learning
- Environment can randomize wall heights/positions on reset (curriculum learning).
- Control these changes in `custom_env.py` (e.g., increase wall height as agent improves).
- Randomize other aspects for sensors to detect as needed.

## 5. Visualization Script (`visualize_custom_env.py`)
- Test and visualize environment changes and sensor outputs.
- Overlays sensor readings (e.g., torso height) on simulation video.
- Extend to visualize your own exteroceptive sensor data.



---
# GitHub Copilot (GPT-4.1) guidance:

## Grade E: Implement Height Map Sensors & Integrate into RL

1. **Add Height Map Sensors (Raycasting)**
	- In `custom_env.py`, implement raycasting from several points on the robot (e.g., feet, torso, or a ring around the base) downwards to measure the distance to the terrain.
	- Collect these distances into a vector (height map) representing the terrain under/around the robot.

2. **Integrate Sensor Data into Observations**
	- Modify the environment’s observation space to include the height map sensor readings alongside the usual proprioceptive data (joint angles, velocities, etc.).
	- Ensure the RL agent receives this new information at every step.

3. **Train the RL Policy**
	- Use your training script (e.g., `custom_ppo_train.py`) to train a policy with the new observation space.
	- The agent should now have access to both internal state and exteroceptive (height map) information.


## Grade C: Demonstrate Policy Adapts to Obstacles

4. **Train and Evaluate**
	- Train the policy in environments with obstacles (walls, steps, rough terrain).
	- Visualize and compare the robot’s behavior with and without the height map sensors.
	- Show that the policy adapts its gait or actions when encountering obstacles (e.g., slows down, lifts legs higher, changes path).

5. **Document Results**
	- Record videos or plots showing the difference in behavior.
	- Briefly explain how the exteroceptive information changes the policy’s response to obstacles.


## Grade A: Teacher-Student Learning with Privileged Information

6. **Teacher Policy (Privileged Information)**
	- Train a “teacher” policy with access to full, noise-free, or privileged information (e.g., perfect terrain map, full robot pose).
	- This policy should perform very well in simulation.

7. **Student Policy (Noisy/Partial Information)**
	- Train a “student” policy with more realistic, noisy, or partial observations (e.g., noisy height map, no access to some state variables).
	- Use imitation learning: have the student mimic the teacher’s actions (behavior cloning, DAgger, or similar).

8. **Compare Direct vs. Teacher-Student Training**
	- Optionally, train a policy directly with the noisy/partial information (no teacher).
	- Compare the performance and learning speed of the student (imitation) vs. direct training.

9. **Analyze and Discuss**
	- Summarize the results: How does privileged information help? What are the challenges with noisy/partial observations?
	- Relate your findings to sim-to-real transfer issues.


**Tips:**
- Start simple: Get the height map sensors working and integrated before moving to imitation learning.
- Use visualization: Scripts like `visualize_custom_env.py` are great for debugging and demonstrating sensor effects.
- Keep experiments manageable: For the teacher-student part, use a small number of episodes and simple imitation (e.g., behavior cloning) to save time.
- Document everything: Take notes and save results for your report.
