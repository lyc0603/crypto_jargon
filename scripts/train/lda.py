"""
Script to train LDA model
"""

import jieba
from gensim.corpora import Dictionary
from gensim.models import LdaModel
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

dictionary = Dictionary(sentences)
corpus = [dictionary.doc2bow(sentence) for sentence in sentences]

# Train LDA model
lda_model = LdaModel(corpus, num_topics=10, id2word=dictionary, minimum_probability=0)


    