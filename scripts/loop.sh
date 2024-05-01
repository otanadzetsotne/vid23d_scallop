#!/bin/bash
set -e

# Check if both arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    exit 1
fi

# Define the input directory from the first argument
input_dir=$1

# Ensure the input directory ends with a slash
[[ "$input_dir" != */ ]] && input_dir="$input_dir/"

# Define the output directory from the second argument
output_dir=$2

# Ensure the output directory ends with a slash
[[ "$output_dir" != */ ]] && output_dir="$output_dir/"

# Loop through all .mp4 files in the input directory
for input_video in "$input_dir"*.mp4; do
  echo "Processing $input_video..."
  PYTORCH_ENABLE_MPS_FALLBACK=1 python /Users/otana/Development/vid23d/main.py \
    --input_video="$input_video" \
    --output_dir="$output_dir" \
    --depth_model=DPT_Hybrid \
    --save_depth
  echo "Finished processing $input_video."
done
