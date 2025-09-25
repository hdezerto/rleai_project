# QUESTIONS

- How to compare performance? Just visualize the gifs?

- Check how to integrate the new measurements into the reward function.

- Fix wall height function (sample_wall_heights).

- Check if this is the right approach:
	- Grade E/C: compare proprioceptive vs exteroceptive both trained noise-FREE
	- Grade A: train teacher with noise-FREE/priviledge info, student with noisy/partial info. Compare this student with a policy trained directly with noisy/partial info.
(this is not what the current code implements)


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
- Integrate this sensor data into the RL training loop (building on Lab 1).

## Grade C:

- Train the policy with the new sensor data.
- Demonstrate that the policy changes its behavior when encountering obstacles (e.g., adapts gait or maneuvers differently).


## Grade A:

- Implement a teacher-student learning approach (as in the referenced paper):
	- The teacher is trained with privileged (full, noise-free) information.
	- The student is trained to mimic the teacher but only has access to noisy or partial information (more realistic for real-world deployment).

- Compare the student’s performance to a policy trained directly with noisy/partial information, highlighting the challenges of sim-to-real transfer.



