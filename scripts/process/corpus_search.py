"""
Script to search for a specific word in the corpus
"""

import pandas as pd
from environ.constants import PROCESSED_DATA_PATH

df_corpus = pd.read_csv(f"{PROCESSED_DATA_PATH}/processed.csv")

word = "BTC"
for index, row in df_corpus.iterrows():
    if word in row['processed']:
        print(row['processed'])
        print("\n")
        continue