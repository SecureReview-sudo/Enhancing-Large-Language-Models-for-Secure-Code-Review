from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    LlamaTokenizer
)

from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, load_dataset
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union, List,Any
import datasets
import os
import torch
import numpy as np
from transformers.models.llama.modeling_llama import LlamaForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()


FINETUNE_MODEL_PATH = ''
DEBUG = False
train_data_file_path = "../dataset/train.jsonl"
test_data_file_path = "../dataset/test.jsonl"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

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
If you judge that there is no security risk, output No Issue.

### Code Change:
"""
prompt2 = """

### Review Comment:
"""

base_model = "CodeLlama-7b-instruct-hf"
if DEBUG:
    model = None
else:
    model = AutoModelForCausalLM.from_pretrained(base_model, use_flash_attention_2=True)
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    add_eos_token=True,
    padding_side="right"
)

tokenizer.pad_token = tokenizer.eos_token  


if isinstance(tokenizer, LlamaTokenizer):
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

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
EOT_TOKEN="</s>"
def generate_and_tokenize_prompt(data_point):
    source = build_instruction_prompt(process_diff(data_point['patch']))
    target = data_point['comment']
    full_prompt = source + target
    tokenized = tokenize(full_prompt)
   
    security_mask = create_security_mask(tokenized['input_ids'], tokenizer, full_prompt)
    
    source_token_ids = tokenizer.encode(source, add_special_tokens=False)
    source_len = len(source_token_ids)
    
    target_backtick_mask = build_backtick_mask_via_offsets(
        target_text=target,
        tokenizer=tokenizer
    )
    full_backtick_mask = [0] * len(tokenized['input_ids'])
    for i, val in enumerate(target_backtick_mask):
        if source_len + i < len(tokenized['input_ids']):
            full_backtick_mask[source_len + i] = val

    tokenized['security_mask'] = security_mask
    tokenized['backtick_mask'] = full_backtick_mask
    
    return tokenized

import random
tokenized_train_data = train_data.map(generate_and_tokenize_prompt, remove_columns=train_data.column_names)
tokenized_eval_data = eval_data.map(generate_and_tokenize_prompt, remove_columns=eval_data.column_names)
for index in random.sample(range(len(tokenized_train_data)), 3):
    print(f"Sample {index} of the training set:")
    print(f"Input IDs: {tokenized_train_data[index]['input_ids']}")
    print(f"Labels: {tokenized_train_data[index]['labels']}")
    print(f"Security Mask: {tokenized_train_data[index]['security_mask']}")
    print(f"Decoded Input: {tokenizer.decode(tokenized_train_data[index]['input_ids'])}")

def loss_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,        security_mask=None, backtick_mask=None,

    )-> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        loss = None
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_bt_mask = backtick_mask[..., 1:].contiguous() if backtick_mask is not None else None
        shift_st_mask = security_mask[..., 1:].contiguous() if security_mask is not None else None
        loss_fct = CrossEntropyLoss(reduction="none", ignore_index=-100)
        loss_tensor = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )  
        batch_size, seq_len, _ = shift_logits.shape
        loss_tensor = loss_tensor.view(batch_size, seq_len)
        valid_mask = (shift_labels != -100).float()

        weighted_loss_tensor = loss_tensor


        factor_bt =2.0
        if shift_bt_mask is not None:
           weighted_loss_tensor = torch.where(
                shift_bt_mask == 1,
                factor_bt * weighted_loss_tensor,
               weighted_loss_tensor
            )

        
        factor_sec = 5.0
        if shift_st_mask is not None:
           weighted_loss_tensor = torch.where(
               shift_st_mask == 1,
               factor_sec * weighted_loss_tensor,
               weighted_loss_tensor
           )

        loss = (weighted_loss_tensor * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss ,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

LlamaForCausalLM.forward = loss_forward
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    # more modules, more training, better performance
)

if not DEBUG:
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.train()

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True


from transformers import DataCollatorForSeq2Seq
class DataCollatorForCausalLMSecurityMaskEnabled(DataCollatorForSeq2Seq):
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: Optional[Any] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors

    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0] else None
        security_masks = [feature["security_mask"] for feature in features] if "security_mask" in features[0] else None
        backtick_masks=[feature["backtick_mask"] for feature in features] if "backtick_mask" in features[0] else None
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                else:
                    raise ValueError("Labels should be a list of integers.")

        if security_masks is not None:
            max_security_mask_length = max(len(l) for l in security_masks)
            if self.pad_to_multiple_of is not None:
                max_security_mask_length = (
                    (max_security_mask_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [0] * (max_security_mask_length - len(feature["security_mask"]))
                if isinstance(feature["security_mask"], list):
                    feature["security_mask"] = (
                        feature["security_mask"] + remainder if padding_side == "right" else remainder + feature["security_mask"]
                    )
                else:
                    raise ValueError("security_mask should be a list of integers.")
        if backtick_masks is not None:
            max_backtick_mask_length = max(len(l) for l in backtick_masks)
            if self.pad_to_multiple_of is not None:
                max_backtick_mask_length = (
                    (max_backtick_mask_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [0] * (max_backtick_mask_length - len(feature["backtick_mask"]))
                if isinstance(feature["backtick_mask"], list):
                    feature["backtick_mask"] = (
                        feature["backtick_mask"] + remainder if padding_side == "right" else remainder + feature["backtick_mask"]
                    )
                else:
                    raise ValueError("backtick_mask should be a list of integers.")
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        return batch
data_collator = DataCollatorForCausalLMSecurityMaskEnabled(
    tokenizer=tokenizer,
    padding=True,
    max_length=None,
    pad_to_multiple_of=None,
    label_pad_token_id=-100,
    return_tensors="pt",
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
    lr_scheduler_type ="cosine",
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
    deepspeed='ds_config.json',
)

def validate_dataset(dataset):
    for idx, item in enumerate(dataset):
        if len(item['input_ids']) != len(item['security_mask']):
            print(f"Mismatch at index {idx}")
        if not all(x in [0,1] for x in item['security_mask']):
            print(f"Invalid mask values at index {idx}")

validate_dataset(tokenized_train_data)
validate_dataset(tokenized_eval_data)
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
