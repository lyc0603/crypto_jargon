"""
Script to generate word embedding using RoBERTa model
"""

import shelve
import random
import time
from scripts.mlm_pretrain import train_dataloader
import torch
from tqdm import tqdm
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import get_pool_emb
from torch.utils.data import DataLoader, Dataset
from bert4torch.snippets import sequence_padding
import os
import json


config_path = "/home/yichen/crypto_jargon/data/RoBERTa-wwm-ext/bert_config.json"
checkpoint_path = "/home/yichen/crypto_jargon/data/RoBERTa-wwm-ext/pytorch_model.bin"
model_saved_path_root = "/home/yichen/crypto_jargon/processed_data/pretrain/model/"
dir_training_data = '/home/yichen/crypto_jargon/processed_data/pretrain/dataset'
task_name = 'roberta'
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"


token_dict, compound_tokens = json.load(
    open("/home/yichen/crypto_jargon/data/RoBERTa-wwm-ext/tokenizer_config.json")
)

pretrained_model = build_transformer_model(
    config_path,
    checkpoint_path,
    segment_vocab_size=0,
    with_mlm=True,
    add_trainer=True,
    compound_tokens=compound_tokens,
).to(device)

if os.path.exists(model_saved_path_root + "model_150.ckpt"):
    pretrained_model.load_weights(model_saved_path_root + "model_150.ckpt")  # 加载模型权重

class Model(BaseModel):
    def __init__(self, pool_method="cls"):
        super().__init__()
        self.pool_method = pool_method
        with_pool = "linear" if pool_method == "pooler" else True
        output_all_encoded_layers = True if pool_method == "first-last-avg" else False
        self.bert = pretrained_model

    def forward(self, token1_ids, token2_ids):
        hidden_state1, pooler1 = self.bert([token1_ids])
        pool_emb1 = get_pool_emb(
            hidden_state1, pooler1, token1_ids.gt(0).long(), self.pool_method
        )

        hidden_state2, pooler2 = self.bert([token2_ids])
        pool_emb2 = get_pool_emb(
            hidden_state2, pooler2, token2_ids.gt(0).long(), self.pool_method
        )

        return torch.cosine_similarity(pool_emb1, pool_emb2)

    def encode(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pooler = self.bert([token_ids])
            attention_mask = token_ids.gt(0).long()
            output = get_pool_emb(
                hidden_state, pooler, attention_mask, self.pool_method
            )
        return output


model = Model().to(device)

#dataloader
class MyDataset(Dataset):
    def __init__(self, file):
        super(MyDataset, self).__init__()
        self.file = file
        self.len = self._get_dataset_length()
        self.db = self._load_data()

    def __getitem__(self, index):
        return self.db[str(index)]

    def __len__(self):
        return self.len

    def _get_dataset_length(self):
        file_record_info = self.file + ".json"
        record_info = json.load(open(file_record_info, "r", encoding="utf-8"))
        return record_info["samples_num"]

    def _load_data(self):
        return shelve.open(self.file)

def collate_fn(batch):
    batch_token_ids, batch_labels = [], []
    for item in batch:
        batch_token_ids.append(item['input_ids'])
        batch_labels.append(item['masked_lm_labels'])

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return [batch_token_ids], batch_labels

def get_train_dataloader():
    while True:
        # prepare dataset
        files_training_data = os.listdir(dir_training_data)
        files_training_data = [file.split(".")[0] for file in files_training_data if "train" in file]
        # 防止使用到正在生成的文件
        files_training_data = [i for i in set(files_training_data) if files_training_data.count(i)==4]
        if files_training_data:
            file_train = random.choice(files_training_data)
            for suffix in [".bak", ".dat", ".dir", ".json"]:
                file_old = os.path.join(dir_training_data, file_train + suffix)
                file_new = os.path.join(dir_training_data, task_name + suffix)
                os.renames(file_old, file_new)
            cur_load_file = file_new.split(".")[0]
            print(cur_load_file)
            train_dataloader = DataLoader(MyDataset(cur_load_file), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            break
        else:
            sleep_seconds = 300
            print(f"No training data! Sleep {sleep_seconds}s!")
            time.sleep(sleep_seconds)
            continue
    return train_dataloader
train_dataloader = get_train_dataloader()


if __name__ == "__main__":

    train_embeddings = []
    for token_ids_list, labels in tqdm(train_dataloader):
        print(token_ids_list, labels)
        for token_ids in token_ids_list:
            train_embeddings.append(model.encode(token_ids))
            pass
        if len(train_embeddings) >= 20:
            break
    train_embeddings = torch.cat(train_embeddings, dim=0).cpu().numpy()
    print(train_embeddings)
