#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ "$SCRIPT_DIR" = "/isaac-sim/memoryPerceiver" ]; then
    # Place your specific instructions here
    echo "SCRIPT_DIR is set to /isaac-sim/memoryPerceiver"
else
    source $SCRIPT_DIR/../isaaclab/bin/activate
    echo "tt"
    mkdir $SCRIPT_DIR/memory_perceiver/main/weights
    mkdir $SCRIPT_DIR/memory_perceiver/main/monitor
    mkdir $SCRIPT_DIR/memory_perceiver/main/experiments
    python_exe="python"
    $python_exe -m pip install torchshow
fi
echo 2