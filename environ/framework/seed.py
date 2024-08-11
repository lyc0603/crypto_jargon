"""
Functions to find the seed keyword
"""

from typing import Any

import numpy as np
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH


def cos_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors
    """

    num = vec1.dot(vec2.T)
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    sim = num / denom
    return sim


def search_pos(corpus: list[Any], word: str) -> list[tuple[int, int]]:
    """
    Find the position of a word in the corpus
    """
    positions = []
    col = 0
    for sentence in corpus:
        if word in sentence:
            row = sentence.index(word)
            pos = (col, row)
            positions.append(pos)
        col += 1
    return positions


def get_row(pos: tuple[int, int]) -> int:
    """
    get word's row index from corpus
    """
    return pos[0]


def get_col(pos: tuple[int, int]) -> int:
    """
    get word's column index from corpus
    """
    return pos[1]


def find_seed_kw(
    corpus_texts: list,
    vocab: list[str],
    emb: np.ndarray,
    seed_vec: np.ndarray,
    save_dir: str,
) -> None:
    """
    Function to find seed keyword
    """

    with open(save_dir, "w", encoding="utf-8") as f:
        for word in tqdm(vocab):
            positions = search_pos(corpus_texts, word)

            if not positions:
                f.write(word + " ")
                f.write("0 ")
                f.write("0\n")
                continue

            sen_vec = []
            word_vec = []
            scores = []

            # find most semantic-related sentence with criminal seed
            for i, position in enumerate(positions):
                sentenceEmb = emb[get_row(position)].copy()
                col = get_col(position)
                if col + 1 >= 255:
                    scores.append(0)
                    continue
                else:
                    wordEmb = sentenceEmb[0][col + 1]
                    sim = cos_sim(wordEmb, seed_vec[0][0])
                    scores.append(sim)
            f.write(word + " ")
            f.write(str(max(scores)) + " ")
            # save the position of the words
            f.write(str(get_row(positions[scores.index(max(scores))])) + "\n")
