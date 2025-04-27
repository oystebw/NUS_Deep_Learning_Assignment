#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go up one level to reach the repo root
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 "$REPO_ROOT/scripts/generate_synthetic_clouds_dataset.py" \
  --input_dir "$REPO_ROOT/images/real_clouds" \
  --input_mask "$REPO_ROOT/images/mask.png" \
  --patch_size 256 256 \
  --crop 21 144 1045 656 \
  --output_dir "$REPO_ROOT/images/synthetic_cloud_patches" \
