#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 "$REPO_ROOT/scripts/generate_real_clouds_dataset.py" \
  --input_dir "$REPO_ROOT/images/real_clouds" \
  --crop 21 144 1045 656 \
  --patch_size 256 256 \
  --output_dir "$REPO_ROOT/images/real_cloud_patches"
