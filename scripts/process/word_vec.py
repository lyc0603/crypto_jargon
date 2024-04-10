"""
Script to generate word vectors from a given text file.
"""

from gensim.models import Word2Vec
import pandas as pd

df_corpus = pd.read_csv("/home/yichen/crypto_jargon/processed_data/processed_cut.csv")
sentences = [doc.split(" ") for doc in df_corpus['text'].map(str)]
model = Word2Vec(sentences, vector_size=768, window=10, workers=40, negative=5, sg=1)
