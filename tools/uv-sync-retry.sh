#!/bin/bash
# Retry uv sync as a unit because uv delegates Git dependencies to Git, whose
# transient transport failures are not covered by UV_HTTP_RETRIES.
set -u

for attempt in 1 2 3; do
    if uv sync "$@"; then
        exit 0
    fi
    if ((attempt == 3)); then
        echo "uv sync failed after ${attempt} attempts" >&2
        exit 1
    fi
    echo "uv sync attempt ${attempt} failed; retrying" >&2
    sleep $((attempt * 5))
done
