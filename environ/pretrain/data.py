"""
Functions for loading data.
"""

import json
import shelve
from torch.utils.data import Dataset
from bert4torch.snippets import sequence_padding
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# load the dataset and convert to tensors
class MyDataset(Dataset):
    """
    Dataset class for loading the pretraining data
    """

    def __init__(self, file: str):
        super(MyDataset, self).__init__()
        self.file = file
        self.len = self._get_dataset_length()
        self.db = self._load_data()

    def __getitem__(self, index: int):
        """
        get the item at the given index
        """
        return self.db[str(index)]

    def __len__(self):
        """
        get the length of the dataset
        """
        return self.len

    def _get_dataset_length(self):
        """
        get the recorded length of the dataset
        """
        file_record_info = self.file + ".json"
        record_info = json.load(open(file_record_info, "r", encoding="utf-8"))
        return record_info["samples_num"]

    def _load_data(self):
        """
        load the data from the file
        """
        return shelve.open(self.file)


def collate_fn(batch: list):
    """
    function to collate the batch
    """
    batch_token_ids, batch_labels = [], []
    for item in batch:
        batch_token_ids.append(item["input_ids"])
        batch_labels.append(item["masked_lm_labels"])

    batch_token_ids = torch.tensor(
        sequence_padding(batch_token_ids), dtype=torch.long, device=device
    )
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return [batch_token_ids], batch_labels