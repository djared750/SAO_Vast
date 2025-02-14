#!/bin/bash

# Cause the script to exit on failure.

# Activate the main virtual environment
. /venv/main/bin/activate

# Install your packages
apt install neovim
pip install https://github.com/vBaiCai/python-pesq/archive/master.zip
pip install torch torchaudio einops soundfile torchsde transformers diffusers
pip install stable-audio-tools
pip install huggingface-hub

# Download some useful files
mkdir -p "${WORKSPACE}/SAO/outputs"
export SAO_OUTPUTS_FOLDER="${WORKSPACE}/SAO/outputs"
wget -P "${WORKSPACE}/SAO" "https://raw.githubusercontent.com/djared750/SAO_Vast/refs/heads/main/SAO_stableAudioTools.py"
wget -P "${WORKSPACE}/SAO" "https://raw.githubusercontent.com/djared750/SAO_Vast/refs/heads/main/SAO_diffusers.py"

# Reload Supervisor
supervisorctl reload
