# Handoff

Read `session_state.md` first. Preserve reservation `SD-69241-apertus-1-5-0`; an empty `GLM_RESERVATION` is only for the separate 152-node DP32 probe. Preserve the complete checkpoint under `526a5c6e...` and the GLM conversion cache. The failed 230/272/273-shard namespaces were deleted after exact verification, reclaiming about 21.7 TiB. Use the checked-in container TOML on `srun`; do not submit the raw squashfs path or inherit compute-node Pyxis variables.
