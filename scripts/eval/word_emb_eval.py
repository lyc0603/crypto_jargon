"""
Script to generate the percision@K
"""

import pandas as pd

from environ.constants import MODEL, PROCESSED_DATA_PATH, REP_JARGONS

df_res = pd.DataFrame(index=range(1, 101, 1))

# load the xlsx file
for model in MODEL:
    df = pd.read_excel(f"{PROCESSED_DATA_PATH}/eval/word_emb_finish/{model}.xlsx")
    df = df[[f"{jargon}_res" for jargon in REP_JARGONS]]

    # calculate the precision@k
    model_res = []
    for i in range(1, 101, 1):
        df_target = df.iloc[:i].copy()
        pk = df_target.sum().sum() / (df_target.shape[0] * df_target.shape[1])
        model_res.append(pk)
    df_res[model] = model_res
