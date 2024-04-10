import pandas as pd
from environ.constants import PROCESSED_DATA_PATH


if __name__ == '__main__':

    vocab = set()

    # read from clean data
    df = pd.read_csv(f"{PROCESSED_DATA_PATH}/processed.csv", encoding="utf8")
    corpus = df["processed"].map(str).to_list()

    # iterate through all characters in the corpus
    for doc in corpus:
        for char in doc:
            vocab.add(char)

    # save new words to new_words.txt
    with open(f"{PROCESSED_DATA_PATH}/vocab.txt", "w") as f:
        for word in vocab:
            f.write(word + "\n")