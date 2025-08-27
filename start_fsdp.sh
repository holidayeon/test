#!/bin/sh
pip install --upgrade pip
python -m venv venv
pip install flash-attn --no-build-isolation
poetry install
accelerate launch --config_file fsdp_config.yaml hf_sft_main.py
