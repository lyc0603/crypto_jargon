import re
import time

import numpy as np
import pandas as pd

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH
from environ.utils.phrase_extraction import extract_phrase

if __name__ == '__main__':

    # read from clean data
    df = pd.read_csv(f"{PROCESSED_DATA_PATH}/processed.csv", encoding="utf8")
    # print(df["processed"].head())
    corpus = df["processed"].to_list()

    t1 = time.time()
    # extract new words using entropy and mutual-information
    result = extract_phrase(corpus, top_k=1000)
    t2 = time.time()
    # print(result[:400])

    # save new words to new_words.txt and sorted with T-score
    with open(f"{PROCESSED_DATA_PATH}/new_words.txt", "w") as f:
        for line in result:
            # f.write(line[0] + " ")
            # f.write(str(line[1]) + "\n")
            f.write(line[0] + "\n")

    print("time:", t2 - t1)