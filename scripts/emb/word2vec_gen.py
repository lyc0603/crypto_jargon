"""
Script to generate embedding for world2vec
"""

import pickle

from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH
from scripts.train.word2vec_train import model_cbow, model_sg, sentences

emb_cbow = [
    [model_cbow.wv[word] for word in text] for text in tqdm(sentences)
]

with open(f"{PROCESSED_DATA_PATH}/emb/cbow_emb.pkl", "wb") as f:
    pickle.dump(emb_cbow, f, protocol=4)

emb_sg = [
    [model_sg.wv[word] for word in text] for text in tqdm(sentences)
]

with open(f"{PROCESSED_DATA_PATH}/emb/sg_emb.pkl", "wb") as f:
    pickle.dump(emb_sg, f, protocol=4)