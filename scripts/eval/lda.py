"""
Evaluate the quality of word embedding
"""

import pandas as pd

from scripts.train.lda import lda_model, dictionary
from environ.constants import PROCESSED_DATA_PATH, REP_JARGONS
from typing import Any

df_lda = pd.DataFrame()


def get_similar_words(word: str, lda_model: Any) -> list:
    """
    Get similar words to the given word
    """
    query_word_id = dictionary.token2id[word]
    query_word_topics = lda_model.get_term_topics(query_word_id)
    most_similar_topic_id = max(query_word_topics, key=lambda x: x[1])[0]
    similar_words = [
        (lda_model.id2word[id], value)
        for id, value in lda_model.get_topic_terms(most_similar_topic_id, topn=101)
        if id != query_word_id
    ]
    similar_words = sorted(similar_words, key=lambda x: x[1], reverse=True)
    similar_words = similar_words[:100]

    return similar_words


if __name__ == "__main__":
    df_lda = pd.DataFrame()

    for jargon in REP_JARGONS:
        try: 
            df_lda[jargon] = [
                candidate for candidate, _ in get_similar_words(jargon, lda_model)
            ]
        except: #pylint: disable=bare-except
            df_lda[jargon] = []
            print(f"Error with {jargon}")
        df_lda[jargon + "_res"] = 0

    df_lda.to_excel(f"{PROCESSED_DATA_PATH}/eval/word_emb/lda.xlsx", index=False)
