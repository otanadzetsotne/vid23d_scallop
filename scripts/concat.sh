#!/bin/bash
set -e

PARTS_DIR=/Users/otana/Development/vid23d/.results/main/result

# Navigate to the directory containing the MP4 files
cd "$PARTS_DIR"

# Generate a sorted list of MP4 files based on the numeric part of the filename
printf "file '%s'\n" $(ls res_*.mp4 | sort -t_ -k4,4n) > filelist.txt

# Use ffmpeg to concatenate the files
ffmpeg -y -f concat -safe 0 -i filelist.txt -c copy full.mp4

# Clean up the file list if you don't need it anymore
#rm filelist.txt

cp full.mp4 full_audio.mp4

ffmpeg -y -i .results/main/main.mp4 -i full.mp4 -map 1:v -map 0:a -c:v copy -c:a copy -shortest full_audio.mp4

ffmpeg -y -i full_audio.mp4 -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 192k full_recodec.mp4
