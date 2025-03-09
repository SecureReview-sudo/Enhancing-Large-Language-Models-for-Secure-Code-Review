import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import re
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import numpy as np
def parse_security_review(report):
    if "Now review this code:" in report:
        report = report.split("Now review this code:")[0]
    sections = ["Security type:", "Description:", "Impact:", "Advice:"]
    pattern = r"(" + "|".join(sections) + r")\s*(.*?)\s*(?=(?:" + "|".join(sections) + r")|$)"
    matches = re.findall(pattern, report, re.DOTALL)

    review = {}
    for match in matches:
        key = match[0].rstrip(":")
        value = match[1].strip()
        review[key] = value

    for section in sections:
        key = section.rstrip(":")
        if key not in review:
            review[key] = None

    return review
def load_model_and_tokenizer():
    model_output_dir =""
    #base_model_path = "deepseek-coder-6.7b-instruct"  
    base_model_path = "CodeLlama-7b-instruct-hf"  
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    base_model.eval()

    model = PeftModel.from_pretrained(
        base_model,
        model_output_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    return model, tokenizer

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

def get_similar_examples(current_patch, examples, security_type, n=1):
    """Find similar examples using BM25"""

    filtered_examples = [ex for ex in examples 
                        if ex['security_type'] == security_type 
                        and ex['patch'] != current_patch]
    
    if len(filtered_examples) < n:
        return filtered_examples[:n]
    
    corpus = [process_diff(ex['patch']) for ex in filtered_examples]
    
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    query = process_diff(current_patch).split()
    scores = bm25.get_scores(query)
    top_n_indices = np.argsort(scores)[-n:][::-1]
    
    return [filtered_examples[i] for i in top_n_indices]

def create_few_shot_prompt(patch, examples, security_type):
    base_prompt = f"""Given a code change with identified security type: {security_type}

Your task is to provide a detailed security review focusing on the following aspects:
1. Description: Explain how this code change could lead to {security_type} issues
2. Impact: Describe the potential security consequences if these issues are not addressed
3. Advice: Provide specific recommendations to resolve the {security_type} concerns
Please not repeat the code change or any part of the input in your response and output in the sample format.


Below is the similar example where {security_type} issues were identified:

"""
    for i, example in enumerate(examples, 1):
        base_prompt += f"Example:\n### Code Change:\n{process_diff(example['patch'])}\n\n"
        base_prompt += f"Security Review:\n{example['comment']}\n\n"
    base_prompt += f"Now review this code:\n### Code Change:\n{patch}\n\nSecurity Review:\n"

    return base_prompt

def generate_review(model, tokenizer, prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs.input_ids,
            max_length=3000,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):]

        if len(response) > 3000:
            response = response[:3000] + "..."
        return response
    except Exception as e:
        print(f"Error generating review: {str(e)}")
        return "Error generating security review"

def process_jsonl_file(test_file, train_file):

    model, tokenizer = load_model_and_tokenizer()
  
    train_examples = []
    with open(train_file, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                if all(k in data for k in ['patch', 'comment', 'security_type']):
                    train_examples.append(data)
            except json.JSONDecodeError:
                continue
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  
                test_data.append(json.loads(line))

    for sample in tqdm(test_data, desc="Processing samples"):
            try:
                data = sample
                current_patch = data.get('patch', '')
                security_type = data.get('Security Type', '')
                
                if not current_patch or not security_type or security_type=="No Issue":
                    continue
                
                similar_examples = get_similar_examples(current_patch, train_examples, security_type)
                if len(similar_examples) < 1:
                    print("Warning: Not enough similar examples found")
                    continue
                
                prompt = create_few_shot_prompt(process_diff(current_patch), similar_examples, security_type)
                review = generate_review(model, tokenizer, prompt)
                parsed_review = parse_security_review(review)
             #   sample["Security Type"] =security_type 
                sample["Description"] = parsed_review.get("Description", "")
                sample["Impact"] = parsed_review.get("Impact", "")
                sample["Advice"] = parsed_review.get("Advice", "")
                print(f"Security Type: {security_type}")
                print(f"Patch: {current_patch}\n")
                print(f"Security Review:\n{review}\n")
                print("-" * 80 + "\n")
        
            except Exception as e:
                print(f"Error processing patch: {str(e)}")
                continue
    with open("result2.jsonl", 'w', encoding='utf-8') as f:
        for sample in test_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    test_file = "result.jsonl"
    train_file = "template.jsonl"
    process_jsonl_file(test_file, train_file)