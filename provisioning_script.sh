#!/bin/bash

# Cause the script to exit on failure.
set -eo pipefail

# Activate the main virtual environment
. /venv/main/bin/activate

# Install your packages
pip install https://github.com/vBaiCai/python-pesq/archive/master.zip
pip install torch torchaudio einops soundfile torchsde transformers diffusers
pip install stable-audio-tools

# Download some useful files
mkdir -p "${WORKSPACE}/SAO/outputs"
export SAO_OUTPUTS_FOLDER="${WORKSPACE}/SAO/outputs"
wget -P "${WORKSPACE}/SAO" "https://github.com/djared750/SAO_Vast/blob/main/SAO_diffusers.py"
wget -P "${WORKSPACE}/SAO" "https://github.com/djared750/SAO_Vast/blob/main/SAO_stableAudioTools.py"

# Reload Supervisor
supervisorctl reload
