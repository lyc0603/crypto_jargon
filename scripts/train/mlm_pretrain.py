"""
Script to pretrain a RoBERTa model on the crypto jargon dataset
"""

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
