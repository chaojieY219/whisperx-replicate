#!/bin/bash

set -e

download() {
  local file_url="$1"
  local destination_path="$2"

  if [ ! -e "$destination_path" ]; then
    wget -O "$destination_path" "$file_url"
  else
      echo "$destination_path already exists. No need to download."
  fi
}

vad_model_dir=models/vad
mkdir -p $vad_model_dir

download $(python3 ./get_vad_model_url.py) "$vad_model_dir/whisperx-vad-segmentation.bin"

cog run python
