# Huggingface;  https://huggingface.co/learn/llm-course/en/chapter12/5?fw=pt#practical-exercise-fine-tune-a-model-with-grpo
import torch
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


dataset = load_dataset("mlabonne/smoltldr")
print(dataset)

model_id = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load LoRA
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=16,
    target_modules="all-linear",
)
model = get_peft_model(model, lora_config)

# Reward function
ideal_length = 50


def reward_len(completions, **kwargs):
    return [-abs(ideal_length - len(completion)) for completion in completions]

# Training arguments
training_args = GRPOConfig(
    output_dir="GRPO",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    max_prompt_length=512,
    max_completion_length=96,
    num_generations=4,
    optim="adamw_8bit",
    num_train_epochs=1,
    bf16=True,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    logging_steps=1,
)

# Trainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset["train"],
)

# Train model
trainer.train()
