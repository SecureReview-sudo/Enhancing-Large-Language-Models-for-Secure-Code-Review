
import copy
import random
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List, Tuple, Union
from peft import LoraConfig, get_peft_model
import torch
import torch.distributed
import transformers
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
from transformers.modeling_outputs import CausalLMOutputWithPast
from datasets import load_dataset
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
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

    examples = [s + t for s, t in zip(sources, targets)]

    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]

    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)


    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)

def create_security_mask(input_ids, tokenizer, full_prompt):
    import re
    pattern = r"### Review Comment:\s*\n\s*Security type:\s*\n?"
    match = re.search(pattern, full_prompt)
    if not match:

        print("\\nSecurity Type:\\n' pattern not found in full_prompt")
        security_mask = [0] * len(input_ids)
        return security_mask
    start_char_idx = match.end()
    end_char_idx = full_prompt.find("\n", start_char_idx)
    if end_char_idx == -1:
        end_char_idx = len(full_prompt)
    tokens_up_to_start = tokenizer.encode(full_prompt[:start_char_idx], add_special_tokens=False)
    start_token_idx = len(tokens_up_to_start)
    tokens_up_to_end = tokenizer.encode(full_prompt[:end_char_idx], add_special_tokens=False)
    end_token_idx = len(tokens_up_to_end)

    security_mask = [0] * len(input_ids)
    for idx in range(start_token_idx+1, end_token_idx+1):
        if idx < len(security_mask):
            security_mask[idx] = 1
        else:
            break

    return security_mask
def find_backtick_pairs(text: str):

    backtick_indices = []
    start_idx = 0
    while True:
        pos = text.find("`", start_idx)
        if pos == -1:
            break
        backtick_indices.append(pos)
        start_idx = pos + 1
    pairs = []
    for i in range(0, len(backtick_indices), 2):
        if i + 1 < len(backtick_indices):
            pairs.append((backtick_indices[i], backtick_indices[i+1]))
    return pairs

def build_backtick_mask_via_offsets(target_text: str, tokenizer):

    bt_pairs = find_backtick_pairs(target_text)
    if not bt_pairs:
   
        encoded_no_pair = tokenizer(target_text, add_special_tokens=False, return_offsets_mapping=False)
        return [0] * len(encoded_no_pair["input_ids"])

    encoded = tokenizer(
        target_text,
        add_special_tokens=False,
        return_offsets_mapping=True
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"] 
    mask = [0] * len(input_ids)
    for (start_char, end_char) in bt_pairs:
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_end > start_char and tok_start < end_char:
                mask[i] = 1
    
    return mask


def train_tokenize_function(examples: Dict, tokenizer) -> Dict:
    """Tokenizing function that adds backtick and security masks."""

    sources = [
        build_instruction_prompt(process_diff(instruction))
        for instruction in examples['patch']
    ]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['comment']]
    data_dict = preprocess(sources, targets, tokenizer)
    backtick_masks = []
    for (inp_ids, source_text, target_text) in zip(
        data_dict['input_ids'],
        sources,
        targets
    ):
        source_token_ids = tokenizer.encode(source_text, add_special_tokens=False)
        source_len = len(source_token_ids)

        tgt_only_mask = build_backtick_mask_via_offsets(
            target_text=target_text,
            tokenizer=tokenizer
        )
        
        full_mask = [0] * len(inp_ids)
        for i, val in enumerate(tgt_only_mask):
            if source_len + i < len(inp_ids):
                full_mask[source_len + i] = val

        backtick_masks.append(full_mask)
    security_masks = []
    for input_ids, source, target in zip(data_dict['input_ids'], sources, targets):
        full_prompt = source + target
        sec_mask = create_security_mask(input_ids, tokenizer, full_prompt)
        security_masks.append(sec_mask)
    data_dict['security_mask'] = security_masks
    data_dict['backtick_mask'] = backtick_masks

    return data_dict



@dataclass
class DataCollatorForSupervisedDataset(object):

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, backtick_masks ,security_masks= tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "backtick_mask","security_mask")
        )
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        backtick_masks = [torch.tensor(x) for x in backtick_masks]
        backtick_masks = torch.nn.utils.rnn.pad_sequence(backtick_masks, batch_first=True, padding_value=0)
        security_masks = [torch.tensor(x) for x in security_masks]
        security_masks = torch.nn.utils.rnn.pad_sequence(security_masks, batch_first=True, padding_value=0)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            backtick_mask=backtick_masks,
            security_mask=security_masks,
        )

class CustomTrainer(Trainer):
       def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        labels = inputs.get("labels")
        security_mask = inputs.get("security_mask", None)
        backtick_mask = inputs.get("backtick_mask", None)  
        outputs = model(**inputs)
        logits = outputs.logits.float()

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_bt_mask = backtick_mask[..., 1:].contiguous() if backtick_mask is not None else None
        shift_st_mask = security_mask[..., 1:].contiguous() if security_mask is not None else None
        loss_fct = CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
        loss_tensor = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )  
        batch_size, seq_len, _ = shift_logits.shape
        loss_tensor = loss_tensor.view(batch_size, seq_len)
        valid_mask = (shift_labels != IGNORE_INDEX).float()
        weighted_loss_tensor = loss_tensor

        factor_bt =2
        if shift_bt_mask is not None:
           weighted_loss_tensor = torch.where(
               shift_bt_mask == 1,
               factor_bt * weighted_loss_tensor,
               weighted_loss_tensor
           )

        factor_sec =5
        if shift_st_mask is not None:
          weighted_loss_tensor = torch.where(
              shift_st_mask == 1,
              factor_sec * weighted_loss_tensor,
              weighted_loss_tensor
          )

        loss = (weighted_loss_tensor * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        
        return (loss, outputs) if return_outputs else loss


def train():
    model = AutoModelForCausalLM.from_pretrained(base_model, use_flash_attention_2=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    train_data_file_path = "../dataset/train.jsonl"
    test_data_file_path  = "../dataset/valid.jsonl"
    train_data = load_dataset("json", data_files=train_data_file_path, split="train")
    eval_data  = load_dataset("json", data_files=test_data_file_path,  split="train")

    train_data = train_data.shuffle(seed=42)
    eval_data  = eval_data.shuffle(seed=42)

    train_dataset = train_data.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=4,
        remove_columns=train_data.column_names,
        desc="Encoding train set",
        fn_kwargs={"tokenizer": tokenizer},
    )
    eval_dataset = eval_data.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=4,
        remove_columns=eval_data.column_names,
        desc="Encoding eval set",
        fn_kwargs={"tokenizer": tokenizer},
    )

 
    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token:", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("Training dataset samples:", len(train_dataset))


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

 
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=20,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
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


    for index in random.sample(range(len(train_dataset)), 3):
        print(f"\n--- Sample {index} ---")
        print("Input IDs:", train_dataset[index]['input_ids'])
        print("Labels:",    train_dataset[index]['labels'])
        print("Backtick Mask:", train_dataset[index]['backtick_mask'])
        print("Decoded Input:", tokenizer.decode(train_dataset[index]['input_ids']))

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.0,
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[early_stopping_callback]
    )

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    trainer.train()

if __name__ == "__main__":

    FINETUNE_MODEL_PATH = ''
    base_model = "deepseek-coder-6.7b-instruct"

    train()
