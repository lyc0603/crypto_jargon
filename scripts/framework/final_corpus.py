"""
Script to generate the final corpus for all models
"""

import jieba

from environ.constants import MAXLEN, PROCESSED_DATA_PATH
from environ.pretrain.mlm_data_gen import corpus, word_segment
from scripts.train.tokenizer import tokenizer

# Initialize jieba
jieba.initialize()
jieba.load_userdict(f"{PROCESSED_DATA_PATH}/new_words.txt")

for texts in corpus():
    for text in texts:
        words = word_segment(text)

        token_ids = [tokenizer._token_start_id]

        for word in words:
            word_tokens = tokenizer.tokenize(text=word)[1:-1]
            word_token_ids = tokenizer.tokens_to_ids(word_tokens)

            token_ids.extend(word_token_ids)

        token_ids.append(tokenizer._token_end_id)

        token_ids = token_ids[:MAXLEN]
        padding_length = MAXLEN - len(token_ids)
        token_ids = token_ids + [tokenizer.pad_token_id] * padding_length

        print(words)
        print(token_ids)
