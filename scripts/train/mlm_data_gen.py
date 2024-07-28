"""
Scripts to generate the dataset for MLM training
"""

import json
import os
import time

import jieba
from bert4torch.tokenizers import Tokenizer, load_vocab

from environ.constants import (DATA_PATH, MAX_FILE_NUM, MAXLEN,
                               PROCESSED_DATA_PATH)
from environ.pretrain.mlm_data_gen import (TrainingDatasetRoBerta, corpus,
                                           word_segment)

# Initialize jieba
jieba.initialize()
jieba.load_userdict(f"{PROCESSED_DATA_PATH}/new_words.txt")

dir_training_data = f"{PROCESSED_DATA_PATH}/pretrain/dataset"

# tokenizer
if os.path.exists(f"{DATA_PATH}/RoBERTa-wwm-ext/tokenizer_config.json"):
    token_dict, compound_tokens = json.load(
        open(f"{DATA_PATH}/RoBERTa-wwm-ext/tokenizer_config.json")
    )
else:
    # load and simplify vocab
    token_dict = load_vocab(
        dict_path=f"{DATA_PATH}/RoBERTa-wwm-ext/vocab.txt",
        simplified=False,
        startswith=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )
    pure_tokenizer = Tokenizer(token_dict.copy(), do_lower_case=True)
    # add all unique words from corpus to ./v2/vocab.txt
    words = [
        line.strip()
        for line in open(
            f"{PROCESSED_DATA_PATH}/vocab.txt", "r", encoding="utf-8"
        ).readlines()
    ]
    user_dict = []
    for w in words:
        if w not in token_dict:
            token_dict[w] = len(token_dict)
            user_dict.append(w)
            print(f"{w} is added to token_dict")
        else:
            print(f"{w} is already in token_dict")
        # if len(user_dict) == NUM_WORDS:
        #     break
    compound_tokens = [pure_tokenizer.encode(w)[0][1:-1] for w in user_dict]
    json.dump(
        [token_dict, compound_tokens],
        open(f"{DATA_PATH}/RoBERTa-wwm-ext/tokenizer_config.json", "w"),
    )

tokenizer = Tokenizer(
    token_dict,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s)
)

# dataset
TD = TrainingDatasetRoBerta(tokenizer, word_segment, sequence_length=MAXLEN)

while True:
    train_file = [
        file
        for file in os.listdir(dir_training_data)
        if ("train_" in file) and ("dat" in file)
    ]

    if len(train_file) < MAX_FILE_NUM:
        record_name = f'{dir_training_data}/train_'+ time.strftime('%Y%m%d%H%M%S', time.localtime())
        TD.process(corpus=corpus(), record_name=record_name)
        time.sleep(1)
        print(f"Generate {record_name}")
    else:
        time.sleep(300)
