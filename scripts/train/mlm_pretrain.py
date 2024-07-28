"""
Script to pretrain a RoBERTa model on the crypto jargon dataset
"""

import sys
sys.path.append("/home/yichen/crypto_jargon")

from environ.pretrain.mlm_pretrain import (
    ModelCheckpoint,
    model,
    train_dataloader,
    STEPS_PER_EPOCH,
    EPOCHS,
)

# save the model
checkpoint = ModelCheckpoint()

# pretrain the model
model.fit(
    train_dataloader,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    callbacks=[checkpoint],
)
