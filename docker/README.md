# Building the Docker Container
NOTE: *We use `docker buildx` instead of `docker build` for these containers*

This directory contains the `Dockerfile` for NeMo-RL Docker images.
You can build two types of images:
- A **release image** (recommended): Contains everything from the hermetic image, plus the nemo-rl source code and pre-fetched virtual environments for isolated workers.
- A **hermetic image**: Includes the base image, cached dependencies, and worker environments populated with third-party packages. Release assembly installs the project and writes runtime readiness markers offline.

`NRL_IMAGE_PROFILE=full` is the Docker default and builds all declared worker
environments. `NRL_IMAGE_PROFILE=apertus` builds six environments for vLLM
generation, Megatron policy training, and rollout controllers. Runtime actor
support is unchanged; other backends require the full image or environment
creation at runtime. The profile selects workers from the same declaration as
the runtime registry, in `nemo_rl/distributed/actor_environments.py`.

The [CSCS launcher](../infra/slurm/cscs/README.md) defaults to `apertus`, checks
content-addressed dependency manifests before reusing a hermetic image, and
records build-stage timings. Final images contain worker import qualification
results at `/opt/nemo-rl-image-qualification.json`. GPU generation, refit, and
checkpoint resume still require distributed qualification before production use.

For detailed instructions on building these images, please see [docs/docker.md](../docs/docker.md).
