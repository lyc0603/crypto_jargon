"""
Script to generate word embedding from the RoBERTa modelß
"""

import json
import pickle

import jieba
import torch
from bert4torch.models import build_transformer_model
from tqdm import tqdm

from environ.constants import DATA_PATH, MAXLEN, PROCESSED_DATA_PATH
from environ.pretrain.mlm_data_gen import corpus, word_segment
from scripts.train.tokenizer import tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize jieba
jieba.initialize()
jieba.load_userdict(f"{PROCESSED_DATA_PATH}/new_words.txt")

cur_load_file = f"{PROCESSED_DATA_PATH}/emb_data/train_20240729101610"
config_path = f"{DATA_PATH}/RoBERTa-wwm-ext/bert_config.json"
checkpoint_path = f"{DATA_PATH}/RoBERTa-wwm-ext/pytorch_model.bin"

token_dict, compound_tokens = json.load(
    open(f"{DATA_PATH}/RoBERTa-wwm-ext/tokenizer_config.json")
)

# Load the pretrain model
model = build_transformer_model(
    config_path,
    checkpoint_path,
    segment_vocab_size=0,
    with_mlm=True,
    add_trainer=True,
    output_all_encoded_layers=True,
    compound_tokens=compound_tokens,
).to(device)
model.load_weights(f"{PROCESSED_DATA_PATH}/pretrain/model/" + "model_20.ckpt")

text_ids = []

for texts in tqdm(corpus()):
    for text in texts:
        words = word_segment(text)

        # add the start and end token
        token_ids = [tokenizer._token_start_id]

        # tokenize the words
        for word in words:
            word_tokens = tokenizer.tokenize(text=word)[1:-1]
            word_token_ids = tokenizer.tokens_to_ids(word_tokens)

            token_ids.extend(word_token_ids)

        token_ids.append(tokenizer._token_end_id)

        # padding
        token_ids = token_ids[:MAXLEN]
        padding_length = MAXLEN - len(token_ids)
        token_ids = token_ids + [tokenizer.pad_token_id] * padding_length
        text_ids.append(token_ids)


emb = []
for text_id in tqdm(text_ids): 
    bert_emb = model.predict(
        [
            torch.tensor([text_id], dtype=torch.long, device=device)
        ]
    )

    bert_emb = bert_emb[0][-2].cpu().detach().numpy()
    emb.append(bert_emb)

with open(f"{PROCESSED_DATA_PATH}/emb/bert_emb.pkl", "wb") as f:
    pickle.dump(emb, f, protocol=4)