import os
import sys

import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

PROMPTFILE = sys.argv[1]
OUTPUTS_FOLDER = os.environ.get("SAO_OUTPUTS_FOLDER")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, MODEL_CONFIG = get_pretrained_model("stabilityai/stable-audio-open-1.0")
SAMPLE_RATE = MODEL_CONFIG["sample_rate"]
SAMPLE_SIZE = MODEL_CONFIG["sample_size"]

try:
    SAO = MODEL.to(DEVICE)
except:
    print("Error: No cuda device found")
    sys.exit(1)


def generate(single_prompt):
    # Set up text and timing conditioning
    conditioning = [{"prompt": single_prompt, "seconds_start": 0, "seconds_total": 10}]

    # Generate stereo audio
    output = generate_diffusion_cond(
        SAO,
        steps=100,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=SAMPLE_SIZE,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=DEVICE,
    )

    # Rearrange audio batch to a single sequence
    output = rearrange(output, "b d n -> d (b n)")

    # Peak normalize, clip, convert to int16, and save to file
    output = (
        output.to(torch.float32)
        .div(torch.max(torch.abs(output)))
        .clamp(-1, 1)
        .mul(32767)
        .to(torch.int16)
        .cpu()
    )
    torchaudio.save(f"{OUTPUTS_FOLDER}/{single_prompt}.wav", output, SAMPLE_RATE)


# Execution block
with open(PROMPTFILE, "r") as p:
    for prompt in p:
        generate(prompt.rstrip())
