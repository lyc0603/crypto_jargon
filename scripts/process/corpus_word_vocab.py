"""
Script to isolate all words and unique characters in the corpus
"""

import jieba
import pandas as pd
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH

# Initialize jieba
jieba.initialize()
jieba.load_userdict(f"{PROCESSED_DATA_PATH}/new_words.txt")

vocab = set()

# read from clean data
df = pd.read_csv(f"{PROCESSED_DATA_PATH}/processed.csv", encoding="utf8")
corpus = df["processed"].map(str).to_list()

# iterate through all characters in the corpus
for doc in corpus:
    for char in doc:
        vocab.add(char)

# iterate through all words in the corpus
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    # get the text data
    text = str(row['processed'])
    # segment the words in the text data
    words = jieba.cut(text)
    # append the segmented words to the list
    for word in words:
        vocab.add(word)

# save new words to new_words.txt
with open(f"{PROCESSED_DATA_PATH}/vocab.txt", "w") as f:
    for word in vocab:
        f.write(word + "\n")