"""
Script to find seed keyword
"""

from torch.utils.data import DataLoader
from tqdm import tqdm
from scripts.train.tokenizer import tokenizer

from environ.constants import PROCESSED_DATA_PATH
from environ.pretrain.data import MyDataset, collate_fn


cur_load_file = f"{PROCESSED_DATA_PATH}/emb_data/train_20240729101610"

train_dataloader = DataLoader(
    MyDataset(cur_load_file),
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
)

for data in tqdm(train_dataloader):
    print(tokenizer.decode(data[0][0][0].tolist()))