import torch
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig, setup_chat_format


dataset = load_dataset("mlabonne/smoltldr")
print(dataset)

model_id = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
special_tokens_dict = {'additional_special_tokens': ['<jeehoonlovesdayeon>', '<dayeonlovesjeehoon>']}
tokenizer.add_special_tokens(special_tokens_dict)

model.resize_token_embeddings(len(tokenizer))

if tokenizer.chat_template is None:
    model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")


# Load LoRA
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=16,
    target_modules=["q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj"],
    modules_to_save=['embed_tokens', 'lm_head']
)
model = get_peft_model(model, lora_config)

# Prepare dataset for SFT
def format_prompts(examples):
    texts = []
    for i in range(len(examples['content'])):
        text = f"<|im_start|>system\nSummarize the following content concisely.<|im_end|>\n<|im_start|>user\n{examples['content'][i]}<|im_end|>\n<|im_start|>assistant\n{examples['summary'][i]}<|im_end|>"
        texts.append(text)
    return {"text": texts}


# Training arguments with FSDP configuration
training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=2,
    max_length=512,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    save_steps=500,
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    max_grad_norm=1.0,
    max_steps=-1,
    warmup_ratio=0.1,
    group_by_length=True,
    lr_scheduler_type="cosine",
    # report_to="tensorboard",
    save_strategy="steps",
    dataloader_pin_memory=False,
)

# SFT Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args,
    processing_class=tokenizer,
)

print(trainer.model.modules)

# Start training
trainer.train()

# Save the model
trainer.save_model("./final_model")
