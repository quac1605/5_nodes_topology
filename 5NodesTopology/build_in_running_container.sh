#!/bin/bash

# Usage: ./build_in_running_container.sh <python_script> <output_executable_name>
# Example: ./build_in_running_container.sh analyze_cluster.py analyze_cluster

set -e

CONTAINER_NAME="clab-century-serf1"
PY_SCRIPT="analyze_cluster.py"
OUT_EXEC="analyze_cluster"

if [[ -z "$PY_SCRIPT" || -z "$OUT_EXEC" ]]; then
  echo "Usage: $0 <python_script> <output_executable_name>"
  exit 1
fi

echo "Copying $PY_SCRIPT to /opt/serfapp/ in container $CONTAINER_NAME"
docker cp "$PY_SCRIPT" "$CONTAINER_NAME":/opt/serfapp/

echo "Installing Python3, venv, pip, and pyinstaller inside the container..."
docker exec "$CONTAINER_NAME" bash -c "apt update && apt install -y python3 python3-venv python3-pip && python3 -m venv /opt/serfapp/venv && /opt/serfapp/venv/bin/pip install --upgrade pip pyinstaller"

echo "Building executable with pyinstaller inside container at /opt/serfapp/..."
docker exec "$CONTAINER_NAME" bash -c "cd /opt/serfapp && /opt/serfapp/venv/bin/pyinstaller --onefile $(basename "$PY_SCRIPT")"

echo "Copying built executable back to host as $OUT_EXEC"
docker cp "$CONTAINER_NAME":/opt/serfapp/dist/"$(basename "$PY_SCRIPT" .py)" "$OUT_EXEC"

echo "Build complete. Executable saved as $OUT_EXEC"
