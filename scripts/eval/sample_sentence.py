"""
Script to retrieve a sample sentence from the dataset
"""

import pandas as pd

from environ.constants import PROCESSED_DATA_PATH
from environ.pretrain.mlm_data_gen import corpus, word_segment

word = "远月供"

df = pd.read_csv(f"{PROCESSED_DATA_PATH}/processed_cut.csv")

for texts in corpus():
    for text in texts:
        words = word_segment(text)
        if word in words:
            print(text)