"""
Evaluate the quality of word embedding
"""

import pandas as pd

from scripts.train.word2vec_train import model_cbow, model_sg
from environ.constants import PROCESSED_DATA_PATH, REP_JARGONS


def get_similar_words(word: str, model) -> pd.DataFrame:
    """
    Get similar words to the given word
    """
    similar_words = model.wv.most_similar(word, topn=100)
    return similar_words


if __name__ == "__main__":
    df_cbow = pd.DataFrame()
    df_sg = pd.DataFrame()

    for jargon in REP_JARGONS:
        df_cbow[jargon] = [
            candidate for candidate, _ in get_similar_words(jargon, model_cbow)
        ]
        df_cbow[jargon + "_res"] = 0
        df_sg[jargon] = [
            candidate for candidate, _ in get_similar_words(jargon, model_sg)
        ]
        df_sg[jargon + "_res"] = 0

    df_cbow.to_excel(f"{PROCESSED_DATA_PATH}/eval/word_emb/cbow.xlsx", index=False)
    df_sg.to_excel(f"{PROCESSED_DATA_PATH}/eval/word_emb/sg.xlsx", index=False)
