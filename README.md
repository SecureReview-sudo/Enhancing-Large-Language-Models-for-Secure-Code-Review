# SecureReviewer: Enhancing Large Language Models for Secure Code Review

## Introduction
Providing personalized and timely feedback for programming assignments is useful for programming education. Automated program repair (APR) techniques have been used to fix the bugs in programming assignments, where the Large Language Models (LLMs) based approaches have shown promising results. Given the growing complexity of identifying and fixing errors in advanced programming assignments, current fine-tuning strategies for APR are inadequate in guiding the LLM to identify errors and make accurate edits during the generative repair process. Furthermore, the autoregressive decoding approach employed by the LLM could potentially impede the efficiency of the repair, thereby hindering the ability to provide timely feedback.

To tackle these challenges, we propose FastFixer, an efficient and effective approach for programming assignment repair. To assist the LLM in accurately identifying and repairing bugs, we first propose a novel repair-oriented fine-tuning strategy, aiming to enhance the LLM's attention towards learning how to generate the necessary patch and its associated context. Furthermore, to speed up the patch generation, we propose an inference acceleration approach that is specifically tailored for the program repair task.

The evaluation results demonstrate that FastFixer obtains an overall improvement of 20.46% in assignment fixing when compared to the state-of-the-art baseline. Considering the repair efficiency, FastFixer achieves a remarkable inference speedup of 16.67× compared to the autoregressive decoding algorithm.

## Setup

### Pre-requisites
- gcc ≥ 9.30
- python ≥ 3.8
- cuda 11.8
- python ≥ 3.9.6

### Install dependencies
```bash
pip install -r requirements.txt
```


### Model Finetuning
1. You can change other hyperparameters in the finetune_codellama file, such as learning rate, batch size, deepspeed hypermeters, etc.
2. Run the following command to finetune the model,Make sure to update the file paths to match your local directory structure:
```bash
PYTHONPATH=. TOKENIZERS_PARALLELISM=false deepspeed --num_gpus=[num of gpus you want to use] finetune/finetune{}.py 
```
### Retrieval-augmented Review Generation
1. You can run the RAG.py script for RAG.
2. Make sure to update the file paths to match your local directory structure.
```bash
python RAG.py
```
### Evaluate
1. You can run the test.py script for inference.
2. Make sure to update the file paths to match your local directory structure.
```bash
python test.py
```

