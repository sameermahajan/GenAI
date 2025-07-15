# Direct Preference Optimization (DPO) implementation

# Warning control
import warnings
warnings.filterwarnings('ignore')

import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()

import torch
import pandas as pd
import tqdm
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset, Dataset
from helper import generate_responses, test_model_with_questions, load_model_and_tokenizer

USE_GPU = False

questions = [
    "What is your name?",
    "Are you ChatGPT?",
    "Tell me about your name and organization."
]

model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct",
                                            USE_GPU)

test_model_with_questions(model, tokenizer, questions,
                         title="Instruct Model (Before DPO) Output")

raw_ds = load_dataset("mrfakename/identity", split="train")

# Show the first 5 elements of the raw dataset
pd.set_option("display.max_colwidth", None)   # show full text in every cell
pd.set_option("display.max_columns", None)    # show all columns
pd.set_option("display.width", 0)             # let the browser handle wrapping

sample_df = raw_ds.select(range(5)).to_pandas()
# print(sample_df)  

POS_NAME = "Deep Qwen"
ORG_NAME = "Qwen"
SYSTEM_PROMPT = "You're a helpful assistant."

if not USE_GPU:
    raw_ds = raw_ds.select(range(5))

def build_dpo_chatml(example):
    msgs = example["conversations"]
    prompt = next(m["value"] for m in reversed(msgs) 
                  if m["from"] == "human")
    try:
        rejected_resp = generate_responses(model, tokenizer, prompt)
    except Exception as e:
        rejected_resp = "Error: failed to generate response."
        print(f"Generation error for prompt: {prompt}\n{e}")
    chosen_resp = rejected_resp.replace(ORG_NAME, POS_NAME)
    chosen = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": chosen_resp},
    ]
    rejected = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": rejected_resp},
    ]

    return {"chosen": chosen, "rejected": rejected}

dpo_ds = raw_ds.map(build_dpo_chatml, remove_columns=raw_ds.column_names)

# dpo_ds = load_dataset("banghua/DL-DPO-Dataset", split="train")

# set up the display configures in pandas
pd.set_option("display.max_colwidth", None)  
pd.set_option("display.width", 0)      


sample_df = dpo_ds.select(range(5)).to_pandas()
# print(sample_df)

config = DPOConfig(
    beta=0.2, 
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_steps=2,
    bf16=False,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=config,    
    processing_class=tokenizer,  
    train_dataset=dpo_ds
)

dpo_trainer.train()

test_model_with_questions(dpo_trainer.model, tokenizer, questions,
                          title="Post-trained Model (After DPO) Output")
