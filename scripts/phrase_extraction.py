"""
Script to extract phrases from a given text.
"""
import math
import re
import types
from collections import Counter
from collections.abc import Iterable
from functools import reduce
from operator import mul

import pandas as pd
from pygtrie import Trie

CPU_COUNT = 1

def union_word_freq(dict1, dict2):
    keys = (dict1.keys() | dict2.keys())
    total = {}
    for key in keys:
        total[key] = dict1.get(key, 0) + dict2.get(key, 0)
    return total

def sentence_split_by_punc(corpus:str):
    return re.split(r'[;；.。，,！\n!?？]',corpus)

def remove_irregular_chars(corpus:str):
    return re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", corpus)

def generate_ngram(corpus,n:int=2):
    def generate_ngram_str(text:str,n):
        # all combinations of n-grams in a text
        for i in range(0, len(text)-n+1):
            yield text[i:i+n]
    if isinstance(corpus,str):
        for ngram in generate_ngram_str(corpus,n):
            yield ngram
    elif isinstance(corpus, (list, types.GeneratorType)):
        for text in corpus:
            for ngram in generate_ngram_str(text,n):
                yield ngram


def get_ngram_freq_info(
        corpus,
        min_n:int=2,
        max_n:int=4,
        chunk_size:int=5000,
        min_freq:int=2,
):
    ngram_freq_total = {}
    # ngram_keys are always in the memory
    ngram_keys = {i: set() for i in range(1, max_n + 2)}

    def _process_corpus_chunk(corpus_chunk):
        ngram_freq = {}

        # generate 1～max_n+1 ngram
        for ni in [1] + list(range(min_n,max_n+2)):
            ngram_generator = generate_ngram(corpus_chunk, ni)
            nigram_freq = dict(Counter(ngram_generator))
            # continue to add ngrams to the set 
            ngram_keys[ni] = (ngram_keys[ni] | nigram_freq.keys())
            # add ngram freq to the dict
            ngram_freq = {**nigram_freq, **ngram_freq}

        # filter out ngrams with freq less than min_freq
        ngram_freq = {word: count for word, count in ngram_freq.items() if count >= min_freq}

        return ngram_freq
    
    if isinstance(corpus,types.GeneratorType):
        ## 注意: 如果corpus是generator, 该function对chunk_size无感知
        for corpus_chunk in corpus:
            ngram_freq = _process_corpus_chunk(corpus_chunk)
            ngram_freq_total = union_word_freq(ngram_freq, ngram_freq_total)
    elif isinstance(corpus, list):
        len_corpus = len(corpus)
        for i in range(0, len_corpus, chunk_size):
            corpus_chunk = corpus[i:min(len_corpus, i + chunk_size)]
            ngram_freq = _process_corpus_chunk(corpus_chunk)
            # merge ngram freqs
            ngram_freq_total = union_word_freq(ngram_freq, ngram_freq_total)
    for k in ngram_keys:
        # filter out ngrams that are not in the ngram_freq_total
        ngram_keys[k] = ngram_keys[k] & ngram_freq_total.keys()
    return ngram_freq_total, ngram_keys

def _ngram_entropy_scorer(parent_ngrams_freq):
    _total_count = sum(parent_ngrams_freq)
    _parent_ngram_probas = map(lambda x: x/_total_count, parent_ngrams_freq)
    _entropy = sum(map(lambda x: -1 * x * math.log(x,2), _parent_ngram_probas))

    return _entropy

def _calc_ngram_entropy(
    ngram_freq,
    ngram_keys,
    n
):
    # calculate entropy for n-grams
    if isinstance(n,Iterable):
        entropy = {}
        for ni in n:
            entropy = {**entropy,**_calc_ngram_entropy(ngram_freq,ngram_keys,ni)}
        return entropy
    
    ngram_entropy = {}
    target_ngrams = ngram_keys[n]
    parent_candidates = ngram_keys[n+1]

    if CPU_COUNT == 1:
        left_neighbors = Trie()
        right_neighbors = Trie()

        for parent_candidate in parent_candidates:
            right_neighbors[parent_candidate] = ngram_freq[parent_candidate]
            left_neighbors[parent_candidate[1:]+parent_candidate[0]] = ngram_freq[parent_candidate]

        for target_ngram in target_ngrams:
            try:  ## 一定情况下, 一个candidate ngram 没有左右neighbor
                right_neighbor_counts = (right_neighbors.values(target_ngram))
                right_entropy = _ngram_entropy_scorer(right_neighbor_counts)
            except KeyError:
                right_entropy = 0
            try:
                left_neighbor_counts = (left_neighbors.values(target_ngram))
                left_entropy = _ngram_entropy_scorer(left_neighbor_counts)
            except KeyError:
                left_entropy = 0
            ngram_entropy[target_ngram] = (left_entropy,right_entropy)
        return ngram_entropy
    else:
        ## TODO 多进程计算
        pass

def _calc_ngram_pmi(ngram_freq,ngram_keys,n):
    """
    计算 Pointwise Mutual Information 与 Average Mutual Information
    :param ngram_freq:
    :param ngram_keys:
    :param n:
    :return:
    """
    if isinstance(n,Iterable):
        mi = {}
        for ni in n:
            mi = {**mi,**_calc_ngram_pmi(ngram_freq,ngram_keys,ni)}
        return mi
    n1_totalcount = sum([ngram_freq[k] for k in ngram_keys[1] if k in ngram_freq])
    target_n_total_count = sum([ngram_freq[k] for k in ngram_keys[n] if k in ngram_freq])
    mi = {}
    for target_ngram in ngram_keys[n]:
        target_ngrams_freq = ngram_freq[target_ngram]
        # ngram / total ngram
        joint_proba = target_ngrams_freq/target_n_total_count
        # char1 / total char1 * char2 / total char2 * ... * charn / total charn
        indep_proba = reduce(mul,[ngram_freq[char] for char in target_ngram])/((n1_totalcount)**n)
        pmi = math.log(joint_proba/indep_proba,2)   #point-wise mutual information
        ami = pmi/len(target_ngram)                 #average mutual information
        mi[target_ngram] = (pmi,ami)
    return mi


# read from clean data
df = pd.read_csv("/home/yichen/crypto_jargon/processed_data/processed.csv", encoding="utf8")
# print(df["processed"].head())
corpus = df["processed"].to_list()

if isinstance(corpus, str):
    corpurs_splits = [remove_irregular_chars(sent) for sent in sentence_split_by_punc(corpus)]
elif isinstance(corpus, list):
    corpus_splits = [
        remove_irregular_chars(sent) for news in corpus for sent in sentence_split_by_punc(str(news)) if len(remove_irregular_chars(sent)) != 0
    ]
else:
    pass

# get scores
min_n = 2
max_n = 4
chunk_size = 1000000
min_freq = 5
top_k = 200

ngram_freq, ngram_keys = get_ngram_freq_info(corpus_splits,min_n,max_n,
                                                 chunk_size=chunk_size,
                                                 min_freq=min_freq)
left_right_entropy = _calc_ngram_entropy(ngram_freq,ngram_keys,range(min_n,max_n+1))
mi = _calc_ngram_pmi(ngram_freq,ngram_keys,range(min_n,max_n+1))
joint_phrase = mi.keys() & left_right_entropy.keys()
word_liberalization = lambda le,re: math.log((le * 2 ** re + re * 2 ** le+0.00001)/(abs(le - re)+1),1.5)
word_info_scores = {word: (mi[word][0],     #point-wise mutual information
                mi[word][1],                   #average mutual information
                left_right_entropy[word][0],   #left_entropy
                left_right_entropy[word][1],   #right_entropy
                min(left_right_entropy[word][0],left_right_entropy[word][1]),    #branch entropy  BE=min{left_entropy,right_entropy}
                word_liberalization(left_right_entropy[word][0],left_right_entropy[word][1])+mi[word][1]   #our score
                    )
            for word in joint_phrase}
new_words = [(item[0], item[1][-1]) for item in sorted(
        word_info_scores.items(), key=lambda item:item[1][-1], reverse=True)]

## DONE 对在candidate ngram中, 首字或者尾字出现次数特别多的进行筛选, 如"XX的,美丽的,漂亮的"剔出字典
target_ngrams = word_info_scores.keys()
start_chars = Counter([n[0] for n in target_ngrams])
end_chars = Counter([n[-1] for n in target_ngrams])
threshold = int(len(target_ngrams) * 0.01)
threshold = max(50,threshold)
invalid_start_chars = set([char for char, count in start_chars.items() if count > threshold])
invalid_end_chars = set([char for char, count in end_chars.items() if count > threshold])

invalid_target_ngrams = set([n for n in target_ngrams if (n[0] in invalid_start_chars or n[-1] in invalid_end_chars)])

for n in invalid_target_ngrams:  ## 按照不合适的字头字尾信息删除一些
    word_info_scores.pop(n)