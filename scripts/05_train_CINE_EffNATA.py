import glob
import os
import random
import sys
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision("high")
random.seed(42)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
src_dir = PROJECT_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd
from hydra import compose, initialize_config_dir

from segment2d import MultiViewDataset, SingleViewBatchSampler, EfficientNATA_Net
from segment2d.model import Segmenter_MultiView

config_dir = PROJECT_ROOT / "config"
saved_model_dir = PROJECT_ROOT / "saved_models"
csv_path = PROJECT_ROOT / "data" / "csv_files"

# Extra datasets merged into a CINE view (same label convention: 1 = LV, 2 = MYO, 3 = RV).
# CMR-MULTI csv (CINE_MULTI_{view}_info.csv) is always used.
EXTRA_CSV = {"SAX": ["ACDC_SAX", "MnM_SAX"]}
# ACDC and MnM are small compared with CMR-MULTI, so their training slices are repeated per epoch
# (validation slices are not repeated).
REPEAT_EXTRA = {"ACDC_SAX": 10, "MnM_SAX": 10}
CINE_CKPT_DIR = saved_model_dir / "CINE_EffNATA"  # EfficientNATA_Net weights (different architecture from saved_models/CINE)


def load_files(cfg_data):
    """Return train/val file lists per view, merging CMR-MULTI, ACDC and MnM using the csv `type` column."""
    train_files, val_files, num_classes_dict, class_weights_dict = {}, {}, {}, {}
    for key in cfg_data.CMR_MULTI.CINE_MULTI.keys():
        train_files[key], val_files[key] = [], []
        for name in [f"CINE_MULTI_{key}"] + EXTRA_CSV.get(key, []):
            csv_file = pd.read_csv(csv_path / f"{name}_info.csv")
            # csv paths are relative to the project root
            paths = csv_file["path"].map(lambda p: str(PROJECT_ROOT / p))  # type: ignore
            train = paths[csv_file["type"] == "train"].tolist()
            repeat = REPEAT_EXTRA.get(name, 1)
            train_files[key] += train * repeat
            val_files[key] += paths[csv_file["type"] == "val"].tolist()
            print(
                f"  {name}: train {len(train)} x{repeat} = {len(train) * repeat} "
                f"| val {(csv_file['type'] == 'val').sum()}"
            )
        num_classes_dict[key] = cfg_data.CMR_MULTI.CINE_MULTI[key].num_classes
        class_weights_dict[key] = cfg_data.CMR_MULTI.CINE_MULTI[key].class_weights

        print(f"Training samples for {key}: {len(train_files[key])} | Validation samples: {len(val_files[key])}")
    return train_files, val_files, num_classes_dict, class_weights_dict


def find_cine_checkpoint():
    """Best-dice CINE_EffNATA checkpoint, falling back to last.ckpt. Returns None if there is none."""
    best = sorted(glob.glob(os.path.join(CINE_CKPT_DIR, "dice_*.ckpt")))
    if best:
        return best[-1]
    last = os.path.join(CINE_CKPT_DIR, "last.ckpt")
    return last if os.path.exists(last) else None


def main():
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg_data = compose(config_name="data_config")
        cfg_train = compose(config_name="train_config")

    print("training CINE (CMR-MULTI + ACDC + MnM)...")
    train_files, val_files, num_classes_dict, class_weights_dict = load_files(cfg_data)

    model = EfficientNATA_Net(in_channels=cfg_train.INPUT_DIM_MODEL, head_classes=num_classes_dict) # type: ignore

    segmenter = Segmenter_MultiView(
        model,
        class_weights_dict=class_weights_dict,
        num_classes_dict=num_classes_dict,
        learning_rate=cfg_train.LEARNING_RATE,
        factor_lr=cfg_train.FACTOR_LR,
        patience_lr=cfg_train.PATIENCE_LR,
        size_augmentation=cfg_data.RESIZE_DIM,
    )

    train_dataset = MultiViewDataset(train_files)
    val_dataset = MultiViewDataset(val_files)
    batch_sampler_train = SingleViewBatchSampler(train_dataset, batch_size=cfg_train.BATCH_SIZE)
    batch_sampler_val = SingleViewBatchSampler(val_dataset, batch_size=cfg_train.BATCH_SIZE, shuffle=False)

    # Define data loaders for the training and validation data
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler_train,
        pin_memory=True,
        num_workers=cfg_train.NUM_WORKERS,
        prefetch_factor=cfg_train.PREFETCH_FACTOR,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=batch_sampler_val,
        num_workers=cfg_train.NUM_WORKERS,
        prefetch_factor=cfg_train.PREFETCH_FACTOR,
    )
    # If wandb_logger is True, create a WandbLogger object
    wandb_logger = (
        WandbLogger(
            project="CardiacSegNATA-Net",
            name="CINE_EffNATA",
            resume=False,
        )
        if cfg_train.LOG_WANDB
        else False
    )
    save_dir = saved_model_dir / "CINE_EffNATA"
    os.makedirs(save_dir, exist_ok=True)
    # Initialize a ModelCheckpoint callback to save the model weights after each epoch
    check_point = ModelCheckpoint(
        save_dir,
        filename="dice_{avg_val_dice:0.4f}",
        monitor="avg_val_dice",
        mode="max",
        save_top_k=cfg_train.SAVE_MODEL_TOP_K,
        verbose=True,
        save_weights_only=True,
        auto_insert_metric_name=False,
        save_last=True,
    )

    # Initialize a LearningRateMonitor callback to log the learning rate during training
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # Initialize a EarlyStopping callback to stop training if the validation dice does not improve for a certain number of epochs
    early_stopping = EarlyStopping(
        monitor="avg_val_dice",
        mode="max",
        patience=cfg_train.PATIENCE_ES,
        verbose=True,
        strict=False,
    )

    # Define a dictionary with the parameters for the Trainer object
    PARAMS_TRAINER = {
        "accelerator": cfg_train.ACCELERATOR,
        "devices": cfg_train.DEVICES,
        "benchmark": True,
        "enable_progress_bar": True,
        # "overfit_batches" :5,
        "logger": wandb_logger,
        "callbacks": [check_point, early_stopping, lr_monitor],
        "log_every_n_steps": 1,
        "num_sanity_val_steps": 3,
        "max_epochs": cfg_train.EPOCHS,
        "precision": cfg_train.PRECISION,

    }

    # If transfer learning is enabled, continue from the best CINE_EffNATA weights (weights only: the optimizer,
    # LR schedule and epoch counter start from scratch)
    if cfg_train.USE_TRANSFER_LEARNING.CINE:
        checkpoint = find_cine_checkpoint()
        if checkpoint is None:
            print(f"no CINE_EffNATA checkpoint found in {CINE_CKPT_DIR}, training CINE_EffNATA from scratch ...")
        else:
            print("using pretrained model for CINE_EffNATA ...")
            print(f"load checkpoint: {checkpoint}")
            segmenter.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"])

    trainer = pl.Trainer(**PARAMS_TRAINER)
    # Train the model using the train and validation data loaders
    trainer.fit(segmenter, train_loader, val_loader)

    if cfg_train.LOG_WANDB:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
