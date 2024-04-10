"""
Script to segment the words in the text data
"""

import jieba
import pandas as pd
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH

jieba.initialize()
jieba.load_userdict(f"{PROCESSED_DATA_PATH}/new_words.txt")

df_corpus = pd.read_csv(f"{PROCESSED_DATA_PATH}/processed.csv")
df_corpus_cut = []

# iterate over the rows in the dataframe
for index, row in tqdm(df_corpus.iterrows(), total=df_corpus.shape[0]):
    # get the text data
    text = str(row['processed'])
    # segment the words in the text data
    words = jieba.cut(text)
    # append the segmented words to the list
    df_corpus_cut.append(' '.join(words))

# create a new dataframe with the segmented words
df_corpus_cut = pd.DataFrame(df_corpus_cut, columns=['text'])
df_corpus_cut.drop_duplicates(subset=['text'], inplace=True)
df_corpus_cut.to_csv(f"{PROCESSED_DATA_PATH}/processed_cut.csv", index=False)