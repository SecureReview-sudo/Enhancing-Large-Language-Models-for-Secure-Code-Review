import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from tqdm import tqdm
import re
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
def build_instruction_prompt(input_text: str):
    diff = input_text
    processed_diff =(diff)
    prompt = prompt1 + processed_diff + prompt2
    return prompt


def load_model_and_tokenizer(model_output_dir, base_model_path):

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

    return tokenizer, model

def generate_prediction(input_text, tokenizer, model, max_length=2000):

    prompt = build_instruction_prompt(process_diff(input_text))
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=max_length
    )
    
    input_ids = inputs['input_ids'].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
    
            do_sample=False,

            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def parse_security_review(report):
    """
    Parses the generated security review report and extracts four main sections.

    Args:
        report (str): The generated security review report.

    Returns:
        dict: A dictionary containing the extracted sections. If a section is missing, its value is None.
    """
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

def validate_and_save(test_file_path, output_file_path, tokenizer, model):
    """
    Processes each sample in the test set, generates security reviews, and saves the results.

    Args:
        test_file_path (str): Path to the input JSONL test file.
        output_file_path (str): Path where the output JSONL will be saved.
        tokenizer: The tokenizer.
        model: The fine-tuned model (including PEFT adapters).
    """
    test_data = []
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  
                test_data.append(json.loads(line))


    for sample in tqdm(test_data, desc="Processing samples"):
        patch = sample.get("patch")
        if not patch:

            sample["Security Type"] = None
            sample["Description"] = None
            sample["Impact"] = None
            sample["Advice"] = None
            continue


        report = generate_prediction(patch, tokenizer, model)
        print(report)  

        parsed_review = parse_security_review(report)

        if parsed_review.get("Security type", "No Issue") is None:
            sample["Security Type"] = "No Issue"
        else:
            sample["Security Type"] = parsed_review.get("Security type", "No Issue")
        sample["Description"] = parsed_review.get("Description", "")
        sample["Impact"] = parsed_review.get("Impact", "")
        sample["Advice"] = parsed_review.get("Advice", "")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for sample in test_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Validation complete. Results saved to {output_file_path}")

def main():
    # Paths to your models and data
    model_output_dir =""
   # base_model_path = "/CodeLlama-7b-instruct-hf" 
    base_model_path = "/deepseek-coder-6.7b-instruct"  
    #base_model_path='llama'
    test_file_path = "test.jsonl"  
    output_file_path = "result.jsonl" 
    tokenizer, model = load_model_and_tokenizer(model_output_dir, base_model_path)
    validate_and_save(test_file_path, output_file_path, tokenizer, model)

if __name__ == "__main__":
    main()
