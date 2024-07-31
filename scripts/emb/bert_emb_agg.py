"""
Script to aggregate the word embedding from the RoBERTa model
"""

import glob
import pickle

from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH

emb_files = glob.glob(f"{PROCESSED_DATA_PATH}/bert_emb/*.pkl")

emb = []

for emb_file in tqdm(emb_files):
    with open(emb_file, "rb") as f:
        emb.append(pickle.load(f))