# Building the Docker Container
NOTE: *We use `docker buildx` instead of `docker build` for these containers*

This directory contains the `Dockerfile` for NeMo-RL Docker images.
You can build two types of images:
- A **release image** (recommended): Contains everything from the hermetic image, plus the nemo-rl source code and pre-fetched virtual environments for isolated workers.
- A **hermetic image**: Includes the base image, cached dependencies, and worker environments populated with third-party packages. Release assembly installs the project and writes runtime readiness markers offline.

The `NRL_ACTORS` build argument selects worker environments by fully qualified
actor name, separated by whitespace. Its empty default builds all declared
workers. The selection is validated against the same declaration as the runtime
registry, in `nemo_rl/distributed/actor_environments.py`. Other workers remain
registered at runtime and require their dependencies to be installed before use.

The [CSCS launcher](../infra/slurm/cscs/README.md) supplies its six-worker
`apertus` selection from a site-local profile. Its Slurm scripts, release receipts,
environment definitions and qualification records live under `infra/slurm/cscs/`.
Shared manifest and worker-check helpers remain in `tools/`. Final images contain worker import qualification
results at `/opt/nemo-rl-image-qualification.json`. GPU generation, refit, and
checkpoint resume still require distributed qualification before production use.

For detailed instructions on building these images, please see [docs/docker.md](../docs/docker.md).
