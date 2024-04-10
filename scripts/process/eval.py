"""
Script to evaluate the performance of the model
"""
import pandas as pd
from environ.constants import PROCESSED_DATA_PATH, SEEDS

df_corpus = pd.read_csv(f"{PROCESSED_DATA_PATH}/processed.csv")

df_k = pd.DataFrame()

for seed in SEEDS:
    topn = 10
    model.wv.most_similar(seed, topn=topn)
    df_k[seed] = [candidate for candidate, _ in model.wv.most_similar(seed, topn=topn)]
        