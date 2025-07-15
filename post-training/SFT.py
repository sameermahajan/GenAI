# Warning control
import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

def generate_responses(model, tokenizer, user_message, system_message=None, 
                       max_new_tokens=100):
    # Format chat using tokenizer's chat template
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    # We assume the data are all single-turn conversation
    messages.append({"role": "user", "content": user_message})
        
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Recommended to use vllm, sglang or TensorRT
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return response

def test_model_with_questions(model, tokenizer, questions, 
                              system_message=None, title="Model Output"):
    print(f"\n=== {title} ===")
    for i, question in enumerate(questions, 1):
        response = generate_responses(model, tokenizer, question, 
                                      system_message)
        print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}\n")

def load_model_and_tokenizer(model_name, use_gpu = False):
    
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if use_gpu:
        model.to("cuda")
    
    if not tokenizer.chat_template:
        tokenizer.chat_template = """{% for message in messages %}
                {% if message['role'] == 'system' %}System: {{ message['content'] }}\n
                {% elif message['role'] == 'user' %}User: {{ message['content'] }}\n
                {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }} <|endoftext|>
                {% endif %}
                {% endfor %}"""
    
    # Tokenizer config
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def display_dataset(dataset):
    # Visualize the dataset 
    rows = []
    for i in range(3):
        example = dataset[i]
        user_msg = next(m['content'] for m in example['messages']
                        if m['role'] == 'user')
        assistant_msg = next(m['content'] for m in example['messages']
                             if m['role'] == 'assistant')
        rows.append({
            'User Prompt': user_msg,
            'Assistant Response': assistant_msg
        })
    
    # Display as table
    df = pd.DataFrame(rows)
    pd.set_option('display.max_colwidth', None)  # Avoid truncating long strings
    print(df)

USE_GPU = False

questions = [
    "Give me an 1-sentence introduction of LLM.",
    "Calculate 1+1-1",
    "What's the difference between thread and process?"
]

model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3-0.6B-Base", USE_GPU)

# base model test
# does not give good responses

test_model_with_questions(model, tokenizer, questions, 
                         title="Base Model (Before SFT) Output")

# load training dataset for SFT
train_dataset = load_dataset("banghua/DL-SFT-Dataset")["train"]
if not USE_GPU:
    train_dataset=train_dataset.select(range(100))

display_dataset(train_dataset)

# SFTTrainer config 
sft_config = SFTConfig(
    learning_rate=8e-5, # Learning rate for training. 
    num_train_epochs=1, #  Set the number of epochs to train the model.
    per_device_train_batch_size=1, # Batch size for each device (e.g., GPU) during training. 
    gradient_accumulation_steps=8, # Number of steps before performing a backward/update pass to accumulate gradients.
    gradient_checkpointing=False, # Enable gradient checkpointing to reduce memory usage during training at the cost of slower training speed.
    logging_steps=2,  # Frequency of logging training progress (log every 2 steps).
    bf16=False,
)

sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset, 
    processing_class=tokenizer,
)
sft_trainer.train()

# test the model after SFT
test_model_with_questions(sft_trainer.model, tokenizer, questions, 
                         title="Base Model (After SFT) Output")

del model, tokenizer
