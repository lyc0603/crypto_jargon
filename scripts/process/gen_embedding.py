"""
Script to generate word embedding using RoBERTa model
"""

from .. import mlm_pretrain
import torch
from tqdm import tqdm


if __name__ == "__main__":

    train_embeddings = []
    for token_ids_list, labels in tqdm(mlm_pretrain.train_dataloader):
        for token_ids in token_ids_list:
            train_embeddings.append(mlm_pretrain.model.encode(token_ids))
        # if len(train_embeddings) >= 20:
        #     break
    train_embeddings = torch.cat(train_embeddings, dim=0).cpu().numpy()
    print('train_embeddings done, start pca training...')
    print(train_embeddings)