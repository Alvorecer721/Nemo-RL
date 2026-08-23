# Handoff

Read `session_state.md` first. Preserve reservation `SD-69241-apertus-1-5-0` and the GLM model-conversion cache. The user authorized removing the complete TP1 training checkpoint under `526a5c6e...` after the replacement TP2 Phase-A launch is staged; never remove the conversion cache. The active experiment is a no-fix `TP2/PP18/ETP1/EP16` save followed by a same-topology fresh-allocation resume on 80 nodes. Use the checked-in container TOML on `srun`; do not submit the raw squashfs path or inherit compute-node Pyxis variables.
