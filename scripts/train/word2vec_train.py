"""
Script to generate word vectors from a given text file.
"""

import jieba
from gensim.models import Word2Vec
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH
from environ.pretrain.mlm_data_gen import corpus, word_segment

# Initialize jieba
jieba.initialize()
jieba.load_userdict(f"{PROCESSED_DATA_PATH}/new_words.txt")

sentences = []
unique_words = set()

for texts in tqdm(corpus()):
    for text in texts:
        words = word_segment(text)
        sentences.append(words)

        for word in words:
            unique_words.add(word)


model_cbow = Word2Vec(
    sentences, vector_size=768, window=10, workers=40, negative=5, sg=0, min_count=1
)
model_sg = Word2Vec(
    sentences, vector_size=768, window=10, workers=40, negative=5, sg=1, min_count=1
)
