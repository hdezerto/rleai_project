# Artifacts

This project keeps a small set of curated artifacts in the repository so the results are easy to inspect from GitHub without rerunning long training jobs.

## Trained Policies

The `policies/` directory contains trained policy snapshots:

| File | Description |
| --- | --- |
| `policies/proprioceptive_policy_and_params.dill` | Proprioceptive PPO baseline policy and parameters. |
| `policies/exteroceptive_policy_and_params.dill` | Exteroceptive PPO policy and parameters. |
| `policies/teacher_policy_and_params.dill` | Privileged teacher policy and parameters. |
| `policies/student_policy_and_params.dill` | Student policy and parameters. |

These files are serialized with `dill`. Only load them from a trusted copy of the repository.

## Evaluation Videos

The curated videos are grouped by policy type:

| Directory | Contents |
| --- | --- |
| `eval_videos/proprioceptive/` | Proprioceptive policy evaluation videos. |
| `eval_videos/exteroceptive/` | Exteroceptive policy evaluation videos. |
| `eval_videos/teacher/` | Teacher policy evaluation videos. |
| `eval_videos/student/` | Student policy evaluation videos. |

Each video name follows `<policy>_velocity_case_<id>.mp4`. The case IDs correspond to the velocity-command cases used by the evaluation helper.

## Generated Outputs

Generated training and evaluation outputs are ignored by default:

- `metrics/`
- `gifs/`
- `videos/`
- `checkpoints/`
- `runs/`
- `wandb/`

`plot_metrics.ipynb` expects generated metric files under `metrics/`, for example `metrics/proprioceptive_metrics.dill`. Those metric files are not included in a fresh clone unless they are regenerated or restored from an external artifact store.

Larger repeated artifacts are best kept outside the main repository history, for example in GitHub Releases or Git LFS.
