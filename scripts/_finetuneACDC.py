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

from segment2d import MultiViewDataset, NATA_Net, SingleViewBatchSampler
from segment2d.model import Segmenter_MultiView

config_dir = PROJECT_ROOT / "config"
saved_model_dir = PROJECT_ROOT / "saved_models"
csv_path = PROJECT_ROOT / "data" / "csv_files"

# Fine-tune the CINE model on ACDC only (SAX view, label convention: 1 = LV, 2 = MYO, 3 = RV).
PRETRAINED_CKPT = saved_model_dir / "CINE" / "last-v2.ckpt"
ACDC_VIEW = "SAX"
ACDC_CSV = "ACDC_SAX"
SAVE_DIR = saved_model_dir / "ACDC"


def load_files():
    """Return the ACDC train/val file lists, keyed by view."""
    csv_file = pd.read_csv(csv_path / f"{ACDC_CSV}_info.csv")
    # csv paths are relative to the project root
    paths = csv_file["path"].map(lambda p: str(PROJECT_ROOT / p))  # type: ignore
    train_files = {ACDC_VIEW: paths[csv_file["type"] == "train"].tolist()}
    val_files = {ACDC_VIEW: paths[csv_file["type"] == "val"].tolist()}
    print(f"{ACDC_CSV}: train {len(train_files[ACDC_VIEW])} | val {len(val_files[ACDC_VIEW])}")
    return train_files, val_files


def main():
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg_data = compose(config_name="data_config")
        cfg_train = compose(config_name="train_config", overrides=["LEARNING_RATE=5e-4", "EPOCHS=5000", "DEVICES=[2]"])

    print("fine-tuning CINE model on ACDC only...")
    train_files, val_files = load_files()

    # Same model as CINE: all views keep their heads so the CINE weights load without any change
    num_classes_dict, class_weights_dict = {}, {}
    for key in cfg_data.CMR_MULTI.CINE_MULTI.keys():
        num_classes_dict[key] = cfg_data.CMR_MULTI.CINE_MULTI[key].num_classes
        class_weights_dict[key] = cfg_data.CMR_MULTI.CINE_MULTI[key].class_weights

    model = NATA_Net(in_channels=cfg_train.INPUT_DIM_MODEL, head_classes=num_classes_dict)

    segmenter = Segmenter_MultiView(
        model,
        class_weights_dict=class_weights_dict,
        num_classes_dict=num_classes_dict,
        learning_rate=cfg_train.LEARNING_RATE,
        factor_lr=cfg_train.FACTOR_LR,
        patience_lr=cfg_train.PATIENCE_LR,
        size_augmentation=cfg_data.RESIZE_DIM,
    )

    # Load the CINE weights (weights only: the optimizer, LR schedule and epoch counter start from scratch)
    print(f"load checkpoint: {PRETRAINED_CKPT}")
    segmenter.load_state_dict(torch.load(PRETRAINED_CKPT, map_location="cpu")["state_dict"])

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
            name="ACDC",
            resume=False,
        )
        if cfg_train.LOG_WANDB
        else False
    )
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    # Initialize a ModelCheckpoint callback to save the model weights after each epoch
    check_point = ModelCheckpoint(
        SAVE_DIR,
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

    # Define a dictionary with the parameters for the Trainer object
    PARAMS_TRAINER = {
        "accelerator": cfg_train.ACCELERATOR,
        "devices": cfg_train.DEVICES,
        "benchmark": True,
        "enable_progress_bar": True,
        "logger": wandb_logger,
        "callbacks": [check_point, lr_monitor],
        "log_every_n_steps": 1,
        "num_sanity_val_steps": 3,
        "max_epochs": cfg_train.EPOCHS,
        "precision": cfg_train.PRECISION,
    }

    trainer = pl.Trainer(**PARAMS_TRAINER)
    # Train the model using the train and validation data loaders
    trainer.fit(segmenter, train_loader, val_loader)

    if cfg_train.LOG_WANDB:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
