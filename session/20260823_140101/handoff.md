# Handoff

Read `session_state.md` first. Do not launch another 80-node async GLM checkpoint run until the synchronous control is terminal-green and the large-run launcher archives Ray/NVRx child logs. Do not delete the historical partial checkpoint or GLM conversion cache. Use the checked-in container TOML on `srun`; do not submit the raw squashfs path or inherit compute-node Pyxis variables.
