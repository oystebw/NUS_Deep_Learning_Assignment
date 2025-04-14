#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go up one level to reach the repo root
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 "$REPO_ROOT/scripts/web_scrapping.py" \
  --start_date 2025-04-12 \
  --end_date 2025-04-12 \
  --output_dir "$REPO_ROOT/images/real_clouds"
