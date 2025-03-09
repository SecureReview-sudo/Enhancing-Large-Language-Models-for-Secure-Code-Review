import json
import re
from tqdm import tqdm
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
keywords_dict = {
   "Race Condition": ["race", "racy"],
    "Buffer Overflow": ["buffer", "overflow", "stack", "strcpy", "strcat", "strtok", "gets", "makepath", "splitpath", "heap", "strlen", "out of memory"],
    "Integer Overflow": ["integer", "overflow", "signedness", "widthness", "underflow"],
    "Improper Access": ["improper", "unauthenticated", "gain access", "permission", "hijack", "authenticate", "privilege", "forensic", "hacker", "root", "URL", "form", "field", "sensitive"],
    "Cross Site Scripting (XSS)": ["crosssite", "CSS", "XSS", "malform", "htmlspecialchar"],
    "Denial of Service (DoS)": ["denial service", "dos", "ddos"],
    "Crash": ["crash", "exception"],
    "Resource Leak": ["leak"],
    "Deadlock": ["deadlock"],
    "SQL Injection": ["SQL", "SQLI", "injection", "ondelete"],
    "Format String": ["format", "string", "printf", "scanf", "sanitize"],
    "Cross Site Request Forgery": ["crosssite", "request forgery", "CSRF", "XSRF", "forged", "cookie", "xhttp"],
    "Encryption": ["encrypt", "decrypt", "password", "cipher", "trust", "checksum", "nonce", "salt", "crypto", "mismatch"],
    "Use After Free": ["use-after-free", "dynamic"],
    "Command Injection": ["command", "exec"],

}

nltk.download('punkt', quiet=True)

input_file_path = ''
output_file_path = ''

matched_records = []
stemmer = PorterStemmer()
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def match_keywords(text, keywords_dict):
    processed_text = preprocess_text(text)
    matches = {}
    
    for category, keywords in keywords_dict.items():
        processed_keywords = [preprocess_text(keyword) for keyword in keywords]
        matched_keywords = []

        for i, keyword in enumerate(keywords):
            processed_keyword = processed_keywords[i]
            if re.search(r'\b' + re.escape(processed_keyword) + r'\b', processed_text, re.IGNORECASE):
                matched_keywords.append(keyword)
        
        if matched_keywords:
            matches[category] = matched_keywords
    return matches

with open(input_file_path, 'r', encoding='utf-8') as infile:
    data = json.load(infile)

for record in tqdm(data, desc="Processing"):
    combined_text = f"{record.get('msg', '')}"
    matches = match_keywords(combined_text, keywords_dict)
    if matches:
        record['matched_keywords'] = matches
        if 'oldf' in record:
            del record['oldf']
        matched_records.append(record)

with open(output_file_path, 'w', encoding='utf-8') as outfile:
    json.dump(matched_records, outfile, indent=4)

