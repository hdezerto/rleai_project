
# TO DO
- Exteroceptive flag is implemented, just implement the actual exteroceptive extraction method.
- Integrate knee collision.
- Train the models.
- Student teacher approach.

In the paper (https://arxiv.org/pdf/2201.08117) they use for exteroceptive: height samples around each foot at multiple radii. In the zoom meeting that was also mentioned.

# QUESTIONS

- How to compare performance? Just visualize the gifs?
Finn: Average feet time to check if the robot lifts more its feet to climb steps.

- Should sample_wall_heights use uniform sampling (increasing upper bound), deterministic linear increase, or increase when the reward increases (the robot's performance improves)?

- For evaluation, should the wall height be fixed or increase? Now it's using the max height (fixed).

- Check the perturbation code in the evaluation cell.

- Check the weight for feet_clearance in the proprioceptive training.

- Check if this is the right approach:
	- Grade E/C: compare proprioceptive vs exteroceptive both trained with noisy/partial info.
	- Grade A: train teacher with noise-FREE/priviledge info, use it to train student with noisy/partial info. Compare this student with a policy trained directly with noisy/partial info (same exteroceptive policy trained in Grade E/C, right?)
(this is what the current code implements)


# NOTES
- Set steps_per_reset to maximum 40 to reduce GPU memory usage and avoid RuntimeError: INTERNAL: cuSolver internal error.
- Changing to 10GB GPU might also work.
- If the above does not works switch to CPU with: ```export JAX_PLATFORM_NAME=cpu```
- Search for 'CHECK' to find places to modify/add code.


# USEFUL COMMANDS:
- `deactivate` to exit the virtual environment. 
- `watch -n 1 nvidia-smi` to monitor GPU usage in real-time (run in a separate terminal).



---
## Grade E:

- Implement simulated height map sensors for the robot to sense obstacles.
	I think here the height map should be in the robot's reference frame (body frame).
- Integrate this sensor data into the RL training loop (building on Lab 1).

## Grade C:

- Train the policy with the new sensor data.
- Demonstrate that the policy changes its behavior when encountering obstacles (e.g., adapts gait or maneuvers differently).


## Grade A:

- Implement a teacher-student learning approach (as in the referenced paper):
	- The teacher is trained with privileged (full, noise-free) information.
	- The student is trained to mimic the teacher but only has access to noisy or partial information (more realistic for real-world deployment).

- Compare the student’s performance to a policy trained directly with noisy/partial information, highlighting the challenges of sim-to-real transfer.



