#!/bin/bash
# Activate the ROS environment and workspace.
set -euo pipefail

# Load the build-time pixi environment.
# Disable nounset while sourcing generated environment scripts.
set +u
source /shell-hook.sh
set -u

echo "========================================"
echo "GroundingDINO ROS 2 Container Starting"
echo "========================================"

if [ -f "${ROS2_WS:-/app/ros2_ws}/install/setup.bash" ]; then
    echo "Sourcing ROS2 workspace..."
    # Source the workspace overlay after the conda environment.
    set +u
    source "${ROS2_WS:-/app/ros2_ws}/install/setup.bash"
    set -u
else
    echo "ERROR: ROS2 workspace not built at ${ROS2_WS:-/app/ros2_ws}/install"
    echo "       The image build should have produced it; this is a build bug."
    exit 1
fi

echo "----------------------------------------"
echo "Environment Information:"
echo "ROS_DISTRO      : ${ROS_DISTRO:-unset}"
echo "ROS_DOMAIN_ID   : ${ROS_DOMAIN_ID:-0 (default)}"
echo "RMW             : ${RMW_IMPLEMENTATION:-default}"
echo "ROS2_WS         : ${ROS2_WS:-/app/ros2_ws}"
echo "PYTHONPATH      : ${PYTHONPATH:-unset}"
# Resolve the current username when available.
run_as=$(id -un 2>/dev/null) || run_as="unknown"
echo "Running as      : ${run_as} (uid=$(id -u) gid=$(id -g))"
echo "----------------------------------------"

# Verify required directories are writable.
for d in /output "${HOME:-}"; do
    if [ -n "$d" ] && [ -d "$d" ] && [ ! -w "$d" ]; then
        owner=$(stat -c '%u:%g' "$d" 2>/dev/null || echo "?:?")
        mode=$(stat -c '%a' "$d" 2>/dev/null || echo "?")
        echo "ERROR: $d is not writable by uid=$(id -u) gid=$(id -g)"
        echo "       (owned by $owner, mode $mode)"
        echo ""
        if [ "${owner%%:*}" = "$(id -u)" ]; then
            echo "  You already own it, so the mode is what refuses the write."
            echo "  On the host:"
            echo "      chmod u+w /path/to/that/dir"
        else
            echo "  The host directory mounted there belongs to a different user."
            echo "  Either chown it on the host:"
            echo "      sudo chown -R \$(id -u):\$(id -g) /path/to/that/dir"
            echo "  or run the container as its owner:"
            echo "      docker run --user ${owner%%:*}:${owner##*:} ..."
        fi
        exit 1
    fi
done

# Check CUDA availability through PyTorch.
python - <<'PY'
import torch
if torch.cuda.is_available():
    print(f"CUDA            : {torch.cuda.get_device_name(0)} (torch {torch.__version__})")
else:
    print(f"CUDA            : NOT AVAILABLE (torch {torch.__version__}) -- "
          "start with --gpus all")
PY

# Verify GroundingDINO resolves to the installed package.
python -c "import groundingdino, sys; p = groundingdino.__file__ or ''; \
    print(f'detector        : {p}'); \
    sys.exit(0) if 'site-packages' in p else \
    sys.exit(f'groundingdino resolved to {p!r}, not the installed package')"

echo "----------------------------------------"
echo ""

exec "$@"
