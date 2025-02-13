# Imports
import json
import os
import sys

import soundfile as sf
import torch
from diffusers import StableAudioPipeline

# Setup
PROMPTFILE = sys.argv[1]
OUTPUTS_FOLDER = os.environ.get("SAO_OUTPUTS_FOLDER")
PIPE = StableAudioPipeline.from_pretrained(
    "stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16
)

try:
    SAO = PIPE.to("cuda")
except:
    print("Error: No cuda device found")
    sys.exit(1)

# set the seed for generator
GENERATOR = torch.Generator("cuda").manual_seed(0)


# Generate function
def generate(prompt, negative_prompt):
    # run the generation
    audio = SAO(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=200,
        audio_end_in_s=30.0,
        num_waveforms_per_prompt=3,
        generator=GENERATOR,
    ).audios

    output = audio[0].T.float().cpu().numpy()
    sf.write(f"{OUTPUTS_FOLDER}/{prompt}.wav", output, SAO.vae.sampling_rate)


# Execution block
with open(PROMPTFILE, "r") as p:
    obj = json.load(p)
    for item in obj:
        generate(item["positive_prompt"], item["negative_prompt"])
