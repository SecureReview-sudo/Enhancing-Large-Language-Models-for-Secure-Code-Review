import os
import json
import torch
from datasets import Dataset, load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()

DEBUG = False
FINETUNE_MODEL_PATH = ''
train_data_file_path ="../dataset/train.jsonl"
test_data_file_path = "../dataset/valid.jsonl"

def process_diff(diff):
    difflines = diff.split("\n")[1:]
    difflines = [line for line in difflines if len(line.strip()) > 0]
    map_dic = {"-": 0, "+": 1, " ": 2}
    def f(s):
        if s in map_dic:
            return map_dic[s]
        else:
            return 2
    labels = [f(line[0]) for line in difflines]
    difflines = [line[1:].strip() for line in difflines]
    inputstr = ""
    for label, line in zip(labels, difflines):
        if label == 1:
            inputstr += "<add>" + line
        elif label == 0:
            inputstr += "<del>" + line
        else:
            inputstr += "<keep>" + line

    return inputstr

train_data = load_dataset("json", data_files=train_data_file_path, split="train")
eval_data = load_dataset("json", data_files=test_data_file_path, split="train")
train_data = train_data.shuffle(seed=42)
eval_data = eval_data.shuffle(seed=42)
prompt1 = """You are a highly capable code reviewer specializing in security assessments. Your primary task is to conduct a comprehensive security review of the provided code changes. Identify and evaluate any potential security weaknesses, and generate a detailed review report that includes the following sections:
1. Security Type //The vulnerability type detected
2. Description //Clearly explain the security issue found in the provided code patch.
3. Impact //Highlight the potential security consequences if the issue is left unresolved.
4. Advice //Offer recommendations for resolving the issue.
If you judge that there is no security risk,only output No Issue.

### Code Change:
"""
prompt2 = """

### Review Comment:
"""

base_model = "CodeLlama-7b-instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    add_eos_token=True,   
    padding_side="right",
  
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 


if DEBUG:
    model = None
else:
    model = AutoModelForCausalLM.from_pretrained(base_model, use_flash_attention_2=True)
def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=2048,
        padding=False,
        return_tensors=None
    )
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = prompt1 + str(process_diff(data_point['patch'])) + prompt2 + str(data_point['comment'])
    return tokenize(full_prompt)
import random
tokenized_train_data = train_data.map(generate_and_tokenize_prompt, remove_columns=train_data.column_names)
tokenized_eval_data = eval_data.map(generate_and_tokenize_prompt, remove_columns=eval_data.column_names)
for index in random.sample(range(len(tokenized_train_data)), 3):
            print(f"Sample {index} of the training set: {tokenized_train_data[index]['input_ids']}, {tokenized_train_data[index]['labels']}.")
            print(f"Sample {index} of the training set: {tokenizer.decode(list(tokenized_train_data[index]['input_ids']))}.")

lora_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']

)

if not DEBUG:
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.train()

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

data_collator=DataCollatorForSeq2Seq( 
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    num_train_epochs=20,
    learning_rate=3e-4,
    per_device_eval_batch_size=1,
    remove_unused_columns=False,
    bf16=True,
    report_to="tensorboard",
    logging_strategy="epoch",
    optim="adamw_torch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    gradient_checkpointing=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    output_dir=FINETUNE_MODEL_PATH,
    save_total_limit=3,
    group_by_length=True,
    deepspeed='configs/ds_config.json',
)

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.0,
)


trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_eval_data,
    args=training_args,
    data_collator=data_collator,
    callbacks=[early_stopping_callback],
)

trainer.train()
