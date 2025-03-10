
import json
import numpy as np
import os
from tqdm import tqdm
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

stemmer = PorterStemmer()

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def compute_embedding(text, model):
    processed_text = preprocess_text(text)
    tokens = processed_text.split()
    word_vectors = [model[token] for token in tokens if token in model]
    if word_vectors:
        embedding = np.mean(word_vectors, axis=0)
    else:
        embedding = np.zeros(model.vector_size)
    return embedding

def compute_embeddings(texts, model):
    embeddings = []
    for text in tqdm(texts, desc="Computing embeddings"):
        embedding = compute_embedding(text, model)
        embeddings.append(embedding)
    return embeddings

def calculate_similarity(cwe_embedding, msg_embedding):
    if len(cwe_embedding) == 0 or len(msg_embedding) == 0:
        return 0.0
    similarity = cosine_similarity([cwe_embedding], [msg_embedding])
    return similarity.flatten()[0]

def save_json_file(data, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving {filename}: {e}")

def create_output_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

cwe_weaknesses = [
    {"Category": "API / Function Errors (CWE-1228)", "Description": "Use of potentially harmful, inconsistent, or obsolete functions (built-in or API) or exposing low-level functionality in a function"},
    {"Category": "Audit / Logging Errors (CWE-1210)", "Description": "Exposing excessive or sensitive data in the log, or logging insufficient security-related information that it cannot be used for forensic purposes"},
    {"Category": "Authentication Errors (CWE-1211)", "Description": "Insufficient effort to verify requester's identification either on a password-based or certification-based system, including bypassing authentication process"},
    {"Category": "Authorization Errors (CWE-1212)", "Description": "Insufficient check of actors' permitted capability before performing an operation on their behalf"},
    {"Category": "Bad Coding Practices (CWE-1006)", "Description": "Careless development or maintenance that leads to security issues e.g., active debug code"},
    {"Category": "Behavioral Problems (CWE-438)", "Description": "Unexpected behaviors by incorrectly written code e.g., infinite loop, missing bracket, misinterpretation of inputs, or a function that does not perform as documented"},
    {"Category": "Business Logic Errors (CWE-840)", "Description": "Improper enforcement of usual features that allow an attacker to manipulate resources e.g., password reset without one-time-password"},
    {"Category": "Communication Channel Errors (CWE-417)", "Description": "Improper handling and verification of incoming and outgoing connections e.g., network connection or file access, which can lead to bypass attack"},
    {"Category": "Complexity Issues (CWE-1226)", "Description": "Various unnecessary complexities in software e.g., excessive parameters, large functions, unconditional branching, deep nesting, circular dependency, or in-loop update"},
    {"Category": "Concurrency Issues (CWE-557)", "Description": "Inconsistency caused by a race condition and incorrect synchronization on the particular resources"},
    {"Category": "Credentials Management Errors (CWE-255)", "Description": "Credential information such as password being stored, used, transported, exposed, or modified in the unsecured ways"},
    {"Category": "Cryptographic Issues (CWE-310)", "Description": "Insufficient use of cryptographic algorithms e.g., inappropriate key length, weak hash function, entropy generator, or signature verification"},
    {"Category": "Key Management Errors (CWE-320)", "Description": "Improper use of cryptographic keys e.g., hard-code key, using a key without checking for expiration, exchange key without verifying actors, or reusing nonce in encryption"},
    {"Category": "Data Integrity Issues (CWE-1214)", "Description": "Insufficient or negligence to verify or signify properties of data e.g., data source, data type, or checksum of incoming and outgoing data"},
    {"Category": "Data Processing Errors (CWE-19)", "Description": "Improper manipulation and handling such as parsing, extracting, and comparing of received data in terms of format, structure, and length"},
    {"Category": "Data Neutralization Issues (CWE-137)", "Description": "Lack of effort to prevent code injection and cross-site scripting from incoming data; and to enforce improper encoding and escaping characters for log and format-sensitive output such as CSV"},
    {"Category": "Documentation Issues (CWE-1225)", "Description": "Missing security-critical information in software document and inconsistency between documentation and actual implementation"},
    {"Category": "File Handling Issues (CWE-1219)", "Description": "Insecure and uncontrolled file path traversal or manipulation, including the use of insecure temporary file"},
    {"Category": "Encapsulation Issues (CWE-1227)", "Description": "Incorrect inheritance of program elements that may access restricted data or methods, not isolating system-dependent functions, or parent class refers children's properties"},
    {"Category": "Error Conditions, Return Values, Status Codes (CWE-389)", "Description": "Unwanted actions around return and exception i.e., sensitive return value, not checking returned value, not handling exception, unknown status code, or silently suppressing errors"},
    {"Category": "Expression Issues (CWE-569)", "Description": "Wrong expression in condition statements i.e., always true or false boolean value, incorrect operators, comparing pointers, or incorrect expression precedence"},
    {"Category": "Handler Errors (CWE-429)", "Description": "Missing or improperly implementing the handling functions of an object e.g., intertwining handler functions, or calling non-reentrant functions in the handler function"},
    {"Category": "Information Management Errors (CWE-199)", "Description": "Leak sensitive information or internal states to unexpected actors via any channels"},
    {"Category": "Initialization and Cleanup Errors (CWE-452)", "Description": "Improper initialization and exit routine e.g., using excessive hard-coded data for initialization or not cleaning up the referenced memory space"},
    {"Category": "Data Validation Issues (CWE-1215)", "Description": "Insufficient validation of non-text data that can lead to security flaws e.g., array index, loop condition, or XML data; including early validation and improper restrictions (whitelist and blacklist) to validate against"},
    {"Category": "Lockout Mechanism Errors (CWE-1216)", "Description": "Overly restrictive of account lockout mechanism after a certain number of failed attempts"},
    {"Category": "Memory Buffer Errors (CWE-1218)", "Description": "Insufficient handling of a buffer that leads to security problems i.e., copy external data without checking the size, incorrect size calculation, buffer overflow-underflow, out-of-bound"},
    {"Category": "Numeric Errors (CWE-189)", "Description": "Improper calculation, conversion, comparison of numbers e.g., integer overflow or underflow, wraparound, truncation, off-by-one, endian, or division by zero; which could yield security problems"},
    {"Category": "Permission Issues (CWE-275)", "Description": "Improper assignment of permissions on file, object, and function during installation or inheritance within a program"},
    {"Category": "Pointer Issues (CWE-465)", "Description": "Improper handling of pointer e.g., accessing, returning, or scaling pointer outside the range, dereferencing the invalid pointer, using fixed pointer address, or accessing and releasing wrong pointer"},
    {"Category": "Privilege Issues (CWE-265)", "Description": "Assigning improper system privileges that violate least privilege principle e.g., non-terminated chroot jail, including incorrectly handling or revoking insufficient privileges"},
    {"Category": "Random Number Issues (CWE-1213)", "Description": "Use predictable algorithm or insufficient entropy for critical random number generator"},
    {"Category": "Resource Locking Problems (CWE-411)", "Description": "Improper control of resource locking e.g., not checking for prior lock, improper degree of resource locking, or deadlock"},
    {"Category": "Resource Management Errors (CWE-399)", "Description": "Improper organization of system in general sense e.g., allowing external influence to access file, class, or code; not releasing resource after use, or using after release; duplicating resource identifier"},
    {"Category": "Signal Errors (CWE-387)", "Description": "Improper handling of signal e.g., race condition triggered by handlers, allowing active handler during sensitive operations, not handling with asynchronous-safe functionality, handling multiple signals in single function"},
    {"Category": "State Issues (CWE-371)", "Description": "Allow the program to enter the unexpected state by multiple causes e.g., letting external actor control system setting, passing mutable objects to an untrusted actor, calling a non-reentrant function that calls other non-reentrant function in nested call"},
    {"Category": "String Errors (CWE-133)", "Description": "Use external format string, incorrectly calculate multi-byte string length, or use wrong string comparison method"},
    {"Category": "Type Errors (CWE-136)", "Description": "Access or converse resources with incompatible types i.e., type confusion error"},
    {"Category": "User Interface Security Issues (CWE-355)", "Description": "Storing sensitive information in the UI layer, having an insufficient warning for unsafe or dangerous actions, having an unsupported or obsolete function in the user interface, allowing Clickjacking"},
    {"Category": "User Session Errors (CWE-1217)", "Description": "Leak data element in wrong session or permitting unexpired session"}
]


model_path = "SO_model.bin"
try:
    word_vect = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Word vectors loaded successfully.")
except Exception as e:
    print(f"Error loading word vectors from {model_path}: {e}")
    exit(1)


cwe_descriptions = [cwe['Description'] for cwe in cwe_weaknesses]
cwe_embeddings = compute_embeddings(cwe_descriptions, word_vect)

msg_data_file = '' #codereviewer dataset
msg_data = read_json_file(msg_data_file)
print(f"Total messages loaded: {len(msg_data)}")

output_dir = 'cwe_train'
create_output_directory(output_dir)


cwe_msg_map = {cwe['Category']: [] for cwe in cwe_weaknesses}


total_saved = 0
cwe_category_counts = {cwe['Category']: 0 for cwe in cwe_weaknesses}

for record in tqdm(msg_data, desc="Processing messages"):
    msg_text = record.get('msg', '')
    msg_embedding = compute_embedding(msg_text, word_vect)
    
    for idx, cwe_embedding in enumerate(cwe_embeddings):
        similarity = calculate_similarity(cwe_embedding, msg_embedding)
        if similarity > 0.7: 
            cwe_category = cwe_weaknesses[idx]['Category']
            cwe_msg_map[cwe_category].append({
                "message": record,
                "similarity": float(similarity)
            })
            total_saved += 1
            cwe_category_counts[cwe_category] += 1

for cwe in cwe_weaknesses:
    category = cwe['Category']
    sorted_msgs = sorted(cwe_msg_map[category], key=lambda x: x['similarity'], reverse=True)
    output_path = os.path.join(output_dir, f"{category.replace('/', '_')}.json")
    save_json_file(sorted_msgs, output_path)


print(f"All {total_saved} imformation.")

for category, count in cwe_category_counts.items():
    print(f"  - {category}: {count} ")
