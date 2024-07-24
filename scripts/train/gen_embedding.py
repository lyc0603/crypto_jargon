"""
Script to generate word embedding using RoBERTa model
"""

from scripts.mlm_pretrain import train_dataloader
import torch
from tqdm import tqdm
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import get_pool_emb
import os
import json


config_path = "/home/yichen/crypto_jargon/data/RoBERTa-wwm-ext/bert_config.json"
checkpoint_path = "/home/yichen/crypto_jargon/data/RoBERTa-wwm-ext/pytorch_model.bin"
model_saved_path_root = "/home/yichen/crypto_jargon/processed_data/pretrain/model/"
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

if __name__ == "__main__":

    train_embeddings = []
    for token_ids_list, labels in tqdm(train_dataloader):
        print(token_ids_list, labels)
        for token_ids in token_ids_list:
            # train_embeddings.append(model.encode(token_ids))
        # if len(train_embeddings) >= 20:
        #     break
    train_embeddings = torch.cat(train_embeddings, dim=0).cpu().numpy()
    print(train_embeddings)
