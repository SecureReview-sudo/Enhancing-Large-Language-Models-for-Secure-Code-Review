import sys, math, re, xml.sax.saxutils
import subprocess
import os
import nltk

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
 
    if (nonorm):
        return s.split()
    if type(s) is not str:
        s = " ".join(s)
 
    for (pattern, replace) in normalize1:
        s = re.sub(pattern, replace, s)
    s = xml.sax.saxutils.unescape(s, {'&quot;': '"'})

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
    "State Management": [
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

# 新增函数：检测是否为单一文本块
def is_single_text_input(data):
    """检测输入是否为单一文本块"""
    prediction_fields = ['Security Type', 'Description', 'Impact', 'Advice']
    has_structured_fields = any(data.get(field, '').strip() for field in prediction_fields if field != 'Description')
    
    # 如果只有Description字段有内容，或者存在comment字段但其他字段为空
    if not has_structured_fields:
        return True
    return False

# 新增函数：处理单一文本输入
def normalize_single_text_input(data):
    """将单一文本输入规范化为结构化格式"""
    comment_text = data.get('comment', '').strip()
    description_text = data.get('Description', '').strip()
    single_text = comment_text if comment_text else description_text
    
    if single_text:
        data['Security Type'] = data.get('Security Type', 'No Issue').strip() or 'No Issue'
        data['Description'] = single_text
        data['Impact'] = data.get('Impact', '').strip()
        data['Advice'] = data.get('Advice', '').strip()
    
    return data

def calculate_weighted_bleu(prediction_dict, reference_dict, is_single_text=False):
    if is_single_text:
        weights = {
            'security_type': 0.0,  
            'description': 1.0, 
            'impact': 0.0,
            'advice': 0.0
        }
    else:
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
        if is_single_text:
            scores['security_type'] = 0.0  # 单一文本时不计算类型分数
        else:
            scores['security_type'] = 100.0 if pred_type == ref_type else 0.0
    except:
        scores['security_type'] = 0.0

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
 
    weighted_score = sum(scores[k] * weights[k] for k in weights.keys())
    scores['weighted_total'] = weighted_score

    return scores

def find_overlapping_matches(text, list1, listk):
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
        
    return matches, matchesk

def process_jsonl(file_path, security_keywords):
    results = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        cnt=0
        for line in f:
            data = json.loads(line)
            is_single_text = is_single_text_input(data)
            if is_single_text:
                data = normalize_single_text_input(data)
            
            security_type1 = data.get('security_type')
            security_type2 = data.get('Security Type')
            
            if security_type1=="No Issue":
                continue
            cnt+=1
            
            try:
                if is_single_text:
                    prediction_dict = {
                        'security_type': data.get('Security Type', 'No Issue'),
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
                    
                    scores = calculate_weighted_bleu(prediction_dict, reference_dict, is_single_text=True)
                    common_keywords = security_keywords.get('Common Keywords', [])
                    all_keywords = common_keywords
                    description = data.get('description', '')
                    impact = data.get('impact', '')
                    advice = data.get('advice', '')
                    Description = data.get('Description', '')
                    Impact = data.get('Impact', '') 
                    Advice = data.get('Advice', '')
                    listd, listdk = find_matches(description, all_keywords)
                    listi, listik = find_matches(impact, all_keywords)
                    lista, listak = find_matches(advice, all_keywords)
                    all_ref_keywords = list(set(listd + listi + lista))
                    all_ref_code_keywords = list(set(listdk + listik + listak))
                    listD, listDk = find_overlapping_matches(Description, all_ref_keywords, all_ref_code_keywords)
                    total_ref_keywords = len(all_ref_keywords)
                    total_ref_code_keywords = len(all_ref_code_keywords)
                    matched_keywords = len(list(set(listD)))
                    matched_code_keywords = len(list(set(listDk)))
                    overlap_ratio_keywords = matched_keywords / total_ref_keywords if total_ref_keywords > 0 else 0
                    overlap_ratio_code = matched_code_keywords / total_ref_code_keywords if total_ref_code_keywords > 0 else 0
                    overlap_ratio = 0.5 * overlap_ratio_keywords + 0.5 * overlap_ratio_code
                    
                    results.append({
                        'security_type': security_type1,
                        'is_single_text': True,
                        'all_ref_keywords': all_ref_keywords,
                        'all_ref_code_keywords': all_ref_code_keywords,
                        'matched_keywords': list(set(listD)),
                        'matched_code_keywords': list(set(listDk)),
                        'overlap_ratio': overlap_ratio,
                        'overlap_ratio_keywords': overlap_ratio_keywords,
                        'overlap_ratio_code': overlap_ratio_code
                    })
                    
                    print(f"Single text input processing:")
                    print(f"  Reference keywords: {all_ref_keywords}")
                    print(f"  Reference code keywords: {all_ref_code_keywords}")
                    print(f"  Matched keywords: {list(set(listD))}")
                    print(f"  Matched code keywords: {list(set(listDk))}")
                    print(f"  Overlap ratio: {overlap_ratio:.3f}")
                    print(f"  BLEU score: {scores['weighted_total']:.3f}")
                    
                    total_bleu.append(scores["weighted_total"]*0.5+overlap_ratio*0.5*100)
                
                
                elif (security_type1 == security_type2) and (security_type1 != 'No Issue'):
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
                    # 原有的类型不匹配处理逻辑
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
            except Exception as e:
                print(f"Error processing: {e}")
                print(data.get('Description', ''))
        
        return results

file_path =

results = process_jsonl(file_path, security_keywords)

print(f"Average BLEU score: {sum(total_bleu)/len(total_bleu)}")
print(f"Total samples processed: {len(total_bleu)}")
