"""
Class to pretrain a RoBERTa model on the crypto jargon dataset
"""

import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from bert4torch.callbacks import Callback
from bert4torch.models import build_transformer_model
from bert4torch.optimizers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from environ.pretrain.data import MyDataset, collate_fn

from environ.constants import BATCH_SIZE, DATA_PATH, PROCESSED_DATA_PATH

device = "cuda" if torch.cuda.is_available() else "cpu"

# Path
dir_training_data = f"{PROCESSED_DATA_PATH}/pretrain/dataset"
config_path = f"{DATA_PATH}/RoBERTa-wwm-ext/bert_config.json"
checkpoint_path = f"{DATA_PATH}/RoBERTa-wwm-ext/pytorch_model.bin"
model_saved_path_root = f"{PROCESSED_DATA_PATH}/pretrain/model/"
TASK_NAME = "roberta"

# Parameters
LEARNING_RATE = 5e-6
WEIGHT_DECAY_RATE = 0.01
NUM_WARMUP_STEPS = 3125
NUM_TRAIN_STEPS = 32750
STEPS_PER_EPOCH = 1048
GRAD_ACCUM_STEPS = 16
EPOCHS = NUM_TRAIN_STEPS * GRAD_ACCUM_STEPS // STEPS_PER_EPOCH


# randomly select a file from the corpus folder and generate a dataloader
def get_train_dataloader(batch_size: int | None = BATCH_SIZE, shuffle: bool = True):
    """
    Function to get the training dataloader
    """

    while True:
        # prepare dataset
        files_training_data = os.listdir(dir_training_data)
        files_training_data = [
            file.split(".")[0] for file in files_training_data if "train" in file
        ]

        # avoid using the generating file
        files_training_data = [
            i for i in set(files_training_data) if files_training_data.count(i) == 4
        ]

        if files_training_data:
            file_train = random.choice(files_training_data)
            for suffix in [".bak", ".dat", ".dir", ".json"]:
                file_old = os.path.join(dir_training_data, file_train + suffix)
                file_new = os.path.join(dir_training_data, TASK_NAME + suffix)
                os.renames(file_old, file_new)
            cur_load_file = file_new.split(".")[0]
            train_dataloader = DataLoader(
                MyDataset(cur_load_file),
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn,
            )
            break
        else:
            sleep_seconds = 300
            print(f"No training data! Sleep {sleep_seconds}s!")
            time.sleep(sleep_seconds)
            continue
    return train_dataloader


train_dataloader = get_train_dataloader()

token_dict, compound_tokens = json.load(
    open(f"{DATA_PATH}/RoBERTa-wwm-ext/tokenizer_config.json")
)

model = build_transformer_model(
    config_path,
    checkpoint_path,
    segment_vocab_size=0,
    with_mlm=True,
    add_trainer=True,
    compound_tokens=compound_tokens,
).to(device)

# weight decay
param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": WEIGHT_DECAY_RATE,
    },
    {
        "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]


class MyLoss(nn.CrossEntropyLoss):
    """
    Custom loss function
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, output, batch_labels):
        """
        Forward pass
        """
        y_preds = output[-1]
        y_preds = y_preds.reshape(-1, y_preds.shape[-1])
        return super().forward(y_preds, batch_labels.flatten())


# define the loss and optimizers
optimizer = optim.Adam(
    optimizer_grouped_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY_RATE
)

# load the model
if os.path.exists(model_saved_path_root + "last_model.ckpt"):
    model.load_weights(model_saved_path_root + "last_model.ckpt")
    print("Loaded the last model.")
if os.path.exists(model_saved_path_root + "last_step.pt"):
    model.load_steps_params(model_saved_path_root + "last_step.pt")
    print("Loaded the last step.")
if os.path.exists(model_saved_path_root + "last_optimizer.pt"):
    state_dict = torch.load(
        model_saved_path_root + "last_optimizer.pt", map_location="cpu"
    )
    optimizer.load_state_dict(state_dict)
    print("Loaded the last optimizer.")

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=NUM_WARMUP_STEPS, num_training_steps=NUM_TRAIN_STEPS
)
model.compile(
    loss=MyLoss(ignore_index=0),
    optimizer=optimizer,
    scheduler=scheduler,
    grad_accumulation_steps=GRAD_ACCUM_STEPS,
)


class ModelCheckpoint(Callback):
    """
    Automatically save the latest model
    """

    def on_dataloader_end(self, logs=None):
        model.train_dataloader.dataset.db.close()
        for suffix in [".bak", ".dat", ".dir", ".json"]:
            file_remove = os.path.join(dir_training_data, TASK_NAME + suffix)
            try:
                os.remove(file_remove)
                print(f"Removed training data {file_remove}.")
            except:
                print(f"Failed to remove training data {file_remove}.")

        # regenerate dataloader
        model.train_dataloader = get_train_dataloader()

    def on_epoch_end(self, global_step, epoch, logs=None):

        # save the model every ten epochs
        if epoch % 10 == 0:
            model.save_weights(model_saved_path_root + f"model_{epoch}.ckpt")

        model.save_weights(model_saved_path_root + "last_model.ckpt")
        model.save_steps_params(model_saved_path_root + "last_step.pt")
        torch.save(optimizer.state_dict(), model_saved_path_root + "last_optimizer.pt")
