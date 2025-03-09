#!/usr/bin/python

'''
This script was adapted from the original version by hieuhoang1972 which is part of MOSES. 
'''

# $Id: bleu.py 1307 2007-03-14 22:22:36Z hieuhoang1972 $

'''Provides:

cook_refs(refs, n=4): Transform a list of reference sentences as strings into a form usable by cook_test().
cook_test(test, refs, n=4): Transform a test sentence as a string (together with the cooked reference sentences) into a form usable by score_cooked().
score_cooked(alltest, n=4): Score a list of cooked test sentences.

score_set(s, testid, refids, n=4): Interface with dataset.py; calculate BLEU score of testid against refids.

The reason for breaking the BLEU computation into three phases cook_refs(), cook_test(), and score_cooked() is to allow the caller to calculate BLEU scores for multiple test sets as efficiently as possible.
'''

import sys, math, re, xml.sax.saxutils
import subprocess
import os
import nltk

# Added to bypass NIST-style pre-processing of hyp and ref files -- wade
nonorm = 0

preserve_case = False
eff_ref_len = "shortest"

normalize1 = [
    ('<skipped>', ''),  # strip "skipped" tags
    (r'-\n', ''),  # strip end-of-line hyphenation and join lines
    (r'\n', ' '),  # join lines
    #    (r'(\d)\s+(?=\d)', r'\1'), # join digits
]
normalize1 = [(re.compile(pattern), replace) for (pattern, replace) in normalize1]

normalize2 = [
    (r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', r' \1 '),  # tokenize punctuation. apostrophe is missing
    (r'([^0-9])([\.,])', r'\1 \2 '),  # tokenize period and comma unless preceded by a digit
    (r'([\.,])([^0-9])', r' \1 \2'),  # tokenize period and comma unless followed by a digit
    (r'([0-9])(-)', r'\1 \2 ')  # tokenize dash when preceded by a digit
]
normalize2 = [(re.compile(pattern), replace) for (pattern, replace) in normalize2]


def normalize(s):
    '''Normalize and tokenize text. This is lifted from NIST mteval-v11a.pl.'''
    # Added to bypass NIST-style pre-processing of hyp and ref files -- wade
    if (nonorm):
        return s.split()
    if type(s) is not str:
        s = " ".join(s)
    # language-independent part:
    for (pattern, replace) in normalize1:
        s = re.sub(pattern, replace, s)
    s = xml.sax.saxutils.unescape(s, {'&quot;': '"'})
    # language-dependent part (assuming Western languages):
    s = " %s " % s
    if not preserve_case:
        s = s.lower()  # this might not be identical to the original
    for (pattern, replace) in normalize2:
        s = re.sub(pattern, replace, s)
    return s.split()


def count_ngrams(words, n=4):
    counts = {}
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def cook_refs(refs, n=4):
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.'''

    refs = [normalize(ref) for ref in refs]
    maxcounts = {}
    for ref in refs:
        counts = count_ngrams(ref, n)
        for (ngram, count) in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram, 0), count)
    return ([len(ref) for ref in refs], maxcounts)


def cook_test(test, item, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.'''
    (reflens, refmaxcounts) = item
    test = normalize(test)
    result = {}
    result["testlen"] = len(test)

    # Calculate effective reference sentence length.

    if eff_ref_len == "shortest":
        result["reflen"] = min(reflens)
    elif eff_ref_len == "average":
        result["reflen"] = float(sum(reflens)) / len(reflens)
    elif eff_ref_len == "closest":
        min_diff = None
        for reflen in reflens:
            if min_diff is None or abs(reflen - len(test)) < min_diff:
                min_diff = abs(reflen - len(test))
                result['reflen'] = reflen

    result["guess"] = [max(len(test) - k + 1, 0) for k in range(1, n + 1)]

    result['correct'] = [0] * n
    counts = count_ngrams(test, n)
    for (ngram, count) in counts.items():
        result["correct"][len(ngram) - 1] += min(refmaxcounts.get(ngram, 0), count)

    return result


def score_cooked(allcomps, n=4, ground=0, smooth=1):
    totalcomps = {'testlen': 0, 'reflen': 0, 'guess': [0] * n, 'correct': [0] * n}
    for comps in allcomps:
        for key in ['testlen', 'reflen']:
            totalcomps[key] += comps[key]
        for key in ['guess', 'correct']:
            for k in range(n):
                totalcomps[key][k] += comps[key][k]
    logbleu = 0.0
    all_bleus = []
    for k in range(n):
        correct = totalcomps['correct'][k]
        guess = totalcomps['guess'][k]
        addsmooth = 0
        if smooth == 1 and k > 0:
            addsmooth = 1
        logbleu += math.log(correct + addsmooth + sys.float_info.min) - math.log(guess + addsmooth + sys.float_info.min)
        if guess == 0:
            all_bleus.append(-10000000)
        else:
            all_bleus.append(math.log(correct + sys.float_info.min) - math.log(guess))

    logbleu /= float(n)
    all_bleus.insert(0, logbleu)

    brevPenalty = min(0, 1 - float(totalcomps['reflen'] + 1) / (totalcomps['testlen'] + 1))
    for i in range(len(all_bleus)):
        if i == 0:
            all_bleus[i] += brevPenalty
        all_bleus[i] = math.exp(all_bleus[i])
    return all_bleus


def bleu(refs, candidate, ground=0, smooth=1):
    refs = cook_refs(refs)
    test = cook_test(candidate, refs)
    return score_cooked([test], ground=ground, smooth=smooth)


def splitPuncts(line):
    return ' '.join(re.findall(r"[\w]+|[^\s\w]", line))


def bleu_fromstr(predictions, golds, rmstop=True):
    predictions = [" ".join(nltk.wordpunct_tokenize(predictions[i])) for i in range(len(predictions))]
    golds = [" ".join(nltk.wordpunct_tokenize(g)) for g in golds]
    if rmstop:
        pypath = os.path.dirname(os.path.realpath(__file__))
        stopwords = open(os.path.join(pypath, "stopwords.txt")).readlines()
        stopwords = [stopword.strip() for stopword in stopwords]
        golds = [" ".join([word for word in ref.split() if word not in stopwords]) for ref in golds]
        predictions = [" ".join([word for word in hyp.split() if word not in stopwords]) for hyp in predictions]
    predictions = [str(i) + "\t" + pred.replace("\t", " ") for (i, pred) in enumerate(predictions)]
    golds = [str(i) + "\t" + gold.replace("\t", " ") for (i, gold) in enumerate(golds)]
    goldMap, predictionMap = computeMaps(predictions, golds)
    bleu = round(bleuFromMaps(goldMap, predictionMap)[0], 2)
    return bleu


def computeMaps(predictions, goldfile):
    predictionMap = {}
    goldMap = {}

    for row in predictions:
        cols = row.strip().split('\t')
        if len(cols) == 1:
            (rid, pred) = (cols[0], '')
        else:
            (rid, pred) = (cols[0], cols[1])
        predictionMap[rid] = [splitPuncts(pred.strip().lower())]

    for row in goldfile:
        (rid, pred) = row.split('\t')
        if rid in predictionMap:  # Only insert if the id exists for the method
            if rid not in goldMap:
                goldMap[rid] = []
            goldMap[rid].append(splitPuncts(pred.strip().lower()))


    return (goldMap, predictionMap)


# m1 is the reference map
# m2 is the prediction map
def bleuFromMaps(m1, m2):
    score = [0] * 5
    num = 0.0

    for key in m1:
        if key in m2:
            bl = bleu(m1[key], m2[key][0])
            score = [score[i] + bl[i] for i in range(0, len(bl))]
            num += 1
    return [s * 100.0 / num for s in score]


security_keywords = {
    "Input Validation": [
        "CSS", "XSS", "malform", "htmlspecialchars",
        "SQL", "SQLI", "input", "validation",
        "command", "exec", "unauthorized", "null",
        "request forgery", "CSRF", "XSRF", "forged", "cookie", "xhttp",
        "sanitize", "escape", "filter", "whitelist", "blacklist", "regex", "pattern", "injection"
    ],
    "Exception Handling": [
        "try", "catch",
        "finally", "throw", "panic", "assert",
        "crash", "exception", "error", "handle", "handing", "null",
        "logging", "stack trace", "recover"
    ],
    "Error and State Management": [
        "denial service", "dos", "ddos", "state", "behavior", "error",
      "fallback", "recover", "resilience", "consistency", "failure", "state", "incorrect", "inconsistent", "expose"
    ],
    "Type and Data Handling": [
        "integer", "overflow", "signedness", "widthness", "underflow",
        "type", "convert", "string", "value",
        "casting", "serialization", "deserialization", "parsing", "byte", "precision"
    ],
    "Resource Management": [
        "memory", "resource", "file descriptor", "leak", "double free",
        "use after free", "allocation", "deallocation", "cleanup", "release","buffer", "overflow", "stack", "strcpy"," strcat", "strtok", "gets", "makepath", "splitpath", "heap", "strlen",
"out of memory","dynamic","finalize", "dispose"
    ],
    "Concurrency": [
        "race", "racy",
        "deadlock", "concurrent", "multiple", "threads", "lock", "condition",
        "synchronization", "inconsistent",
        "mutex", "atomic", "semaphore", "critical section", "thread safety", "parallel", "volatile"
    ],
    "Access Control and Information Security": [
        "improper", "unauthenticated", "access", "permission", "sensitive", "information", "protected",
        "hijack", "authenticate", "privilege", "forensic", "hacker", 
        "root", "URL", "form", "field", "leak", "unauthorized",
        "encrypt", "decrypt", "password", "cipher", "trust", "checksum", "nonce", "salt", "crypto", "mismatch", "expose",
        "authorization", "authentication", "role-based", "RBAC", "credential", "session", "token",  "patch", "SSL", "TLS", "certificate"
    ]
,
    "Common Keywords": [
        "security", "vulnerability", "vulnerable", "hole", "exploit", "malicious",
        "attack", "bypass", "backdoor", "threat", "expose", "breach", 
        "violate", "fatal", "blacklist", "overrun", "insecure", "lead",
        "scare", "scary", "conflict", "trojan", "firewall", "spyware", "empty",
        "adware", "virus", "ransom", "malware", "malicious", 
        "dangling", "unsafe", "worm", "phishing", "cve", "cwe", "injection",
        "collusion", "covert", "mitm", "sniffer", "quarantine", "risk","error",
      "spam", "spoof", "tamper", "zombie", "cast", "xml","concern","sensitive","exposure","undefined","insecure", "vulnerability"
    ]
}
import json
import re
total_bleu=[]
def extract_code_words(text):
    pattern = r'`([^`]+)`'
    matches = re.findall(pattern, text)
    result=[]
    for item in matches:
        a="`"+item+'`'
        result.append(a)
    return result

from nltk.corpus import wordnet

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))
import re
from nltk.stem import WordNetLemmatizer

from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import re
import nltk

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN  

def lemmatize_word(word, lemmatizer):
    pos = pos_tag([word])[0][1]
    wn_pos = get_wordnet_pos(pos)
    return lemmatizer.lemmatize(word, pos=wn_pos).lower()

def find_matches(text, keywords):
    lemmatizer = WordNetLemmatizer()
    matches = []
    text = text.replace(',', '').replace('.', '')
    text_words = text.split()
    for word in text_words:
        word_lemma = lemmatize_word(word, lemmatizer)
        if word_lemma in keywords and word_lemma not in matches:
            matches.append(word_lemma)
    
    code_words = extract_code_words(text)
    return matches, code_words
def calculate_weighted_bleu(prediction_dict, reference_dict):
    weights = {
        'security_type': 0.3,
        'description': 0.3,
        'impact': 0.2,
        'advice': 0.2
    }
    
    scores = {}

    try:
        pred_type = prediction_dict.get('security_type', '').strip().lower()
        ref_type = reference_dict.get('security_type', '').strip().lower()
        scores['security_type'] = 100.0 if pred_type == ref_type else 0.0
    except:
        print(prediction_dict.get('description'))
    for key in ['description', 'impact', 'advice']:
        try:
            pred = prediction_dict.get(key, '').strip()
            ref = reference_dict.get(key, '').strip()
            
            if pred and ref: 
                predictions = [pred]
                references = [ref]
                score = bleu_fromstr(predictions, references, rmstop=False)
                scores[key] = score
  
            else:
                scores[key] = 0.0
        except:
            scores[key] = 0.0
            print(prediction_dict)
 
    weighted_score = sum(scores[k] * weights[k] for k in weights.keys())
    scores['weighted_total'] = weighted_score

    return scores
def find_overlapping_matches(text, list1,listk):
    matches = []
    for keyword in list1:
        all_forms = [keyword] + get_synonyms(keyword)
        for form in all_forms:
     
            pattern = r'\b' + re.escape(form) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(keyword)
                break
    matchesk=[]
    for word in listk:
           if '`' in word:
            clean_word = word.replace('`', '')
            for text_word in extract_code_words(text):
                if clean_word in text_word or text_word in clean_word:
                    matchesk.append(word)
                    break
        
    return matches,matchesk
def process_jsonl(file_path, security_keywords):
    results = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        cnt=0
        for line in f:
            data = json.loads(line)
            
            security_type1 = data.get('security_type')
            security_type2 = data.get('Security Type')
            if security_type1=="No Issue":
                continue
            cnt+=1
            try:
                if (security_type1 == security_type2) and (security_type1 != 'No Issue'):
                    prediction_dict = {
            
                    'security_type': data.get('Security Type', ''),
                    'description': data.get('Description', ''),
                    'impact': data.get('Impact', ''),
                    'advice': data.get('Advice', '')
                }
                    reference_dict = {
            
                        'security_type': data.get('security_type', ''),
                        'description': data.get('description', ''),
                        'impact': data.get('impact', ''),
                        'advice': data.get('advice', '')
                    }
                    scores = calculate_weighted_bleu(prediction_dict, reference_dict)
                    type_keywords = security_keywords.get(security_type1, [])
                    common_keywords = security_keywords.get('Common Keywords', [])
                    all_keywords = type_keywords + common_keywords
                    
                    description = data.get('description', '')
                    Description = data.get('Description', '')
                    impact=data.get('impact', '')
                    Impact=data.get('Impact', '')
                    advice=data.get('advice','')
                    Advice=data.get('Advice','')
                    listd ,listdk= find_matches(description, all_keywords)
                    listi,listik=(find_matches(impact,all_keywords))
                    lista,listak=(find_matches(advice,all_keywords))
                    listd=list(set(listd))
                    listi=list(set(listi))
                    lista=list(set(lista))   
                    listdk=list(set(listdk))
                    listik=list(set(listik))
                    listak=list(set(listak))        
                    listD,listDk = find_overlapping_matches(Description, listd,listdk)
                    listI,listIk=find_overlapping_matches(Impact,listi,listik)
                    listA,listAk=find_overlapping_matches(Advice,lista,listak)
                    listD=list(set(listD))
                    listI=list(set(listI))
                    listA=list(set(listA))
                    listDk=list(set(listDk))
                    listIk=list(set(listIk))
                    listAk=list(set(listAk))
                    overlap_ratiod= len(listD) / len(listd) if listd else 0
                    overlap_ratioi = len(listI) / len(listi) if listi else 0
                    overlap_ratioa = len(listA) / len(lista) if lista else 0
                    overlap_ratiodk= len(listDk) / len(listdk) if listdk else 0
                    overlap_ratioik = len(listIk) / len(listik) if listik else 0
                    overlap_ratioak = len(listAk) / len(listak) if listak else 0
                    
                    overlap_ratio = 0.2*(overlap_ratiod+overlap_ratiodk) + 0.15*(overlap_ratioi+overlap_ratioik) + 0.15*(overlap_ratioa+overlap_ratioak)                    
                    results.append({
                        'security_type': security_type1,
                        'listd': listd,
                        'listD': listD,
                        'listi': listi,
                        'listI': listI,
                        'lista': lista,
                        'listA': listA,
                        'listdk': listdk,
                        'listDk': listDk,
                        'listik': listik,
                        'listIk': listIk,
                        'listak': listak,
                        'listAk': listAk,
                        'overlap_ratio': overlap_ratio
                    })
                    print({
                        'security_type': security_type1,
                        'security_type2':security_type2,
                        'listd': listd,
                        'listD': listD,
                        'listi': listi,
                        'listI': listI,
                        'lista': lista,
                        'listA': listA,
                        'listdk': listdk,
                        'listDk': listDk,
                        'listik': listik,
                        'listIk': listIk,
                        'listak': listak,
                        'listAk': listAk,
                        'overlap_ratio': overlap_ratio
                    })
                    total_bleu.append(scores["weighted_total"]*0.5+overlap_ratio*0.5*100)


                elif (security_type1 != security_type2) and (security_type2 != 'No Issue'):
                    prediction_dict = {
                    'security_type': data.get('Security Type', ''),
                    'description': data.get('Description', ''),
                    'impact': data.get('Impact', ''),
                    'advice': data.get('Advice', '')
                }
                    reference_dict = {
                        'security_type': data.get('security_type', ''),
                        'description': data.get('description', ''),
                        'impact': data.get('impact', ''),
                        'advice': data.get('advice', '')
                    }
                    scores = calculate_weighted_bleu(prediction_dict, reference_dict)
                    common_keywords = security_keywords.get('Common Keywords', [])
                
                    all_keywords = common_keywords
                    
                    description = data.get('description', '')
                    Description = data.get('Description', '')
                    impact=data.get('impact', '')
                    Impact=data.get('Impact', '')
                    advice=data.get('advice','')
                    Advice=data.get('Advice','')
                    print(all_keywords)
                    listd ,listdk= find_matches(description, all_keywords)
                    listi,listik=(find_matches(impact,all_keywords))
                    lista,listak=(find_matches(advice,all_keywords))
                    listd=list(set(listd))
                    listi=list(set(listi))
                    lista=list(set(lista))   
                    listdk=list(set(listdk))
                    listik=list(set(listik))
                    listak=list(set(listak))        
                    listD,listDk = find_overlapping_matches(Description, listd,listdk)
                    listI,listIk=find_overlapping_matches(Impact,listi,listik)
                    listA,listAk=find_overlapping_matches(Advice,lista,listak)
                    listD=list(set(listD))
                    listI=list(set(listI))
                    listA=list(set(listA))
                    listDk=list(set(listDk))
                    listIk=list(set(listIk))
                    listAk=list(set(listAk))
                    overlap_ratiod= len(listD) / len(listd) if listd else 0
                    overlap_ratioi = len(listI) / len(listi) if listi else 0
                    overlap_ratioa = len(listA) / len(lista) if lista else 0
                    overlap_ratiodk= len(listDk) / len(listdk) if listdk else 0
                    overlap_ratioik = len(listIk) / len(listik) if listik else 0
                    overlap_ratioak = len(listAk) / len(listak) if listak else 0
                
                    overlap_ratio = 0.2*(overlap_ratiod+overlap_ratiodk) + 0.15*(overlap_ratioi+overlap_ratioik) + 0.15*(overlap_ratioa+overlap_ratioak)                    
                    results.append({
                        'security_type': security_type1,
                        'security_type2':security_type2,
                        'listd': listd,
                        'listD': listD,
                        'listi': listi,
                        'listI': listI,
                        'lista': lista,
                        'listA': listA,
                        'listdk': listdk,
                        'listDk': listDk,
                        'listik': listik,
                        'listIk': listIk,
                        'listak': listak,
                        'listAk': listAk,
                        'overlap_ratio': overlap_ratio
                    })
                    print({
                        'security_type': security_type1,
                        'security_type2':security_type2,
                        'listd': listd,
                        'listD': listD,
                        'listi': listi,
                        'listI': listI,
                        'lista': lista,
                        'listA': listA,
                        'listdk': listdk,
                        'listDk': listDk,
                        'listik': listik,
                        'listIk': listIk,
                        'listak': listak,
                        'listAk': listAk,
                        'overlap_ratio': overlap_ratio
                    })
                    total_bleu.append(scores["weighted_total"]*0.5+overlap_ratio*0.5*100)
        
                elif (security_type1!="No Issue") and (security_type2 == 'No Issue'):
                    total_bleu.append(0)
            except:
                print(data.get('Description', ''))
        
        return results
file_path ='results.jsonl'

results = process_jsonl(file_path, security_keywords)

print(sum(total_bleu)/len(total_bleu))
print(len(total_bleu))