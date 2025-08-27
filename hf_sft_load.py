import torch
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig, setup_chat_format


# Test loading the saved model =================================================
print("Testing model loading...")
from peft import PeftModel
model_id = "Qwen/Qwen3-0.6B"

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load the fine-tuned adapter
loaded_model = PeftModel.from_pretrained(base_model, "./final_model")

# Load tokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained("./final_model")

# Test inference
test_text = "<|im_start|>system\nSummarize the following content concisely.<|im_end|>\n<|im_start|>user\nArtificial Intelligence has revolutionized many industries, from healthcare to finance. Machine learning algorithms can now diagnose diseases, predict market trends, and automate complex tasks that previously required human expertise.<|im_end|>\n<|im_start|>assistant\n"

inputs = loaded_tokenizer(test_text, return_tensors="pt").to(loaded_model.device)
with torch.no_grad():
    outputs = loaded_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=loaded_tokenizer.eos_token_id
    )

generated_text = loaded_tokenizer.decode(outputs[0], skip_special_tokens=False)
print("Generated text:")
print(generated_text)
