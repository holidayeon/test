#!/bin/sh
pip install --upgrade pip setuptools wheel
pip install datasets==3.2.0 transformers==4.51.3 trl==0.14.0 peft==0.14.0 accelerate==1.2.1 bitsandbytes==0.45.2
pip install flash-attn --no-build-isolation
