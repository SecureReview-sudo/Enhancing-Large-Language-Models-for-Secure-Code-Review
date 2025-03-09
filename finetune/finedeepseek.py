import copy
import random
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:512'
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from peft import LoraConfig, get_peft_model
import torch
import torch.distributed
import transformers
from transformers import Trainer
from datasets import load_dataset

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()

IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"

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
def build_instruction_prompt(instruction: str):
    return '''You are a highly capable code reviewer specializing in security assessments. Your primary task is to conduct a comprehensive security review of the provided code changes. Identify and evaluate any potential security weaknesses, and generate a detailed review report that includes the following sections:
1. Security Type //The vulnerability type detected
2. Description //Clearly explain the security issue found in the provided code patch.
3. Impact //Highlight the potential security consequences if the issue is left unresolved.
4. Advice //Offer recommendations for resolving the issue.
If you judge that there is no security risk, output No Issue.

### Code Change:
{}
### Review Comment:
'''.format(instruction.strip()).lstrip()



def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer):
    sources = [ 
        build_instruction_prompt(process_diff(instruction))
        for instruction in examples['patch']
    ]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['comment']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def train():
    from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)


    base_model="deepseek-coder-6.7b-instruct"
    FINETUNE_MODEL_PATH = ''
    model = AutoModelForCausalLM.from_pretrained(base_model, use_flash_attention_2=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    lora_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']

)

 
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.train()

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    training_args=TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    num_train_epochs=20,
    learning_rate=3e-4,
    lr_scheduler_type ="cosine",
    per_device_eval_batch_size=1,
    remove_unused_columns=False,
    bf16=True,
    report_to="tensorboard",
    logging_strategy="epoch",
    optim="adamw_torch",
    eval_strategy="epoch",
    save_strategy="epoch",
    gradient_checkpointing=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    output_dir=FINETUNE_MODEL_PATH,
    save_total_limit=3,
    group_by_length=True,
    deepspeed='ds_config_zero3.json',
)

    train_data_file_path = "../dataset/train.jsonl"
    test_data_file_path ="../dataset/test.jsonl"
    train_data = load_dataset("json", data_files=train_data_file_path, split="train")
    eval_data = load_dataset("json", data_files=test_data_file_path, split="train")
    train_data = train_data.shuffle(seed=42)
    eval_data = eval_data.shuffle(seed=42)    
    train_dataset = train_data.map(
        train_tokenize_function,
        batched=True,
        batch_size=300,
        num_proc=32,
        remove_columns=train_data.column_names,
        desc="Running Encoding",
        fn_kwargs={ "tokenizer": tokenizer }
    )
    eval_dataset = eval_data.map(
        train_tokenize_function,
        batched=True,
        batch_size=20,
        num_proc=32,
        remove_columns=train_data.column_names,
        desc="Running Encoding",
        fn_kwargs={ "tokenizer": tokenizer }
    )
    if 1:
        torch.distributed.barrier()
    
    if 1:
        print("Training dataset samples:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
            print(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")
    early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=4,
    early_stopping_threshold=0.0,
)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator,callbacks=[early_stopping_callback])

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    trainer.train()
if __name__ == "__main__":
    train()
