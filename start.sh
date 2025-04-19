#!/bin/sh
pip install --yes --upgrade pip setuptools wheel
pip install --yes datasets==3.2.0 transformers==4.47.1 trl==0.14.0 peft==0.14.0 accelerate==1.2.1 bitsandbytes==0.45.2
pip install --yes flash-attn --no-build-isolation
