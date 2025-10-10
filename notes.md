
# TO DO
- Review _get_exteroceptive and _get_grid_mean. Should we change to a rectangle instead of a square?
- I think we should change to raw points instead of mean. This allows the robot to identify where the edge of the step is.
- Integrate knee collision. (CHECK IF NECESSARY)
- Tune the models.
- Student teacher approach. (WORKING ON THIS...)

Grade C: Use 3 grids (front and to the sides) for exteroceptive input.
Teacher: Use the same 3 grids, noise-free (privileged information).
Student: Use only the front grid, with noise (realistic/limited information).

 
In the paper (https://arxiv.org/pdf/2201.08117) they use for exteroceptive: height samples around each foot at multiple radii. In the zoom meeting that was also mentioned.


# QUESTIONS

- Is the knee collision implemented? Should we keep it?
- Check the weight for feet_clearance in the proprioceptive training.
- I see prints in the step function but not in the reset function.


# NOTES
- Set steps_per_reset to maximum 40 to reduce GPU memory usage and avoid RuntimeError: INTERNAL: cuSolver internal error.
- Changing to 10GB GPU might also work.
- If the above does not works switch to CPU with: ```export JAX_PLATFORM_NAME=cpu```


Original repo link: https://github.com/finnBsch/eai2025_rl_final

# USEFUL COMMANDS:
- `deactivate` to exit the virtual environment. 
- `watch -n 1 nvidia-smi` to monitor GPU usage in real-time (run in a separate terminal).




