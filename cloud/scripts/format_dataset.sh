#!/bin/bash

# This script formats the synthetic and real cloud patches into
# trainA/trainB/testA/testB for CycleGAN training

# Get absolute path to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
IMAGE_DIR="$PROJECT_ROOT/images"

python3 "$SCRIPT_DIR/dataset_formatting.py" \
  --input_dir "$IMAGE_DIR" \
  --A synthetic_cloud_patches \
  --B real_cloud_patches \
  --folders trainA trainB testA testB \
  --alpha 0.9
