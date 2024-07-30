"""
Script to generate word embedding from the RoBERTa modelß
"""

import json
import pickle

import torch
from bert4torch.models import build_transformer_model
from torch.utils.data import DataLoader
from tqdm import tqdm

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH
from environ.pretrain.data import MyDataset, collate_fn

device = "cpu"

cur_load_file = f"{PROCESSED_DATA_PATH}/emb_data/train_20240729101610"
config_path = f"{DATA_PATH}/RoBERTa-wwm-ext/bert_config.json"
checkpoint_path = f"{DATA_PATH}/RoBERTa-wwm-ext/pytorch_model.bin"

train_dataloader = DataLoader(
    MyDataset(cur_load_file),
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
)

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
model.load_weights(f"{PROCESSED_DATA_PATH}/pretrain/model/" + "model_10.ckpt")

for idx, instances in enumerate(tqdm(train_dataloader)):

    bert_emb = model.predict(
        [
            torch.tensor([instances[0][0][0].tolist()], dtype=torch.long, device=device)
        ]
    )

    bert_emb = [i.cpu().detach().numpy()for i in bert_emb[0]]
    with open(f"{PROCESSED_DATA_PATH}/bert_emb/emb_{idx}.pkl", "wb") as f:
        pickle.dump(bert_emb, f, protocol=pickle.HIGHEST_PROTOCOL)
