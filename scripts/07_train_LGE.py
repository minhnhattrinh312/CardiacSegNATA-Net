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

from segment2d import MultiViewDataset, NATA_Net, SingleViewBatchSampler
from segment2d.model import Segmenter_MultiView

config_dir = PROJECT_ROOT / "config"
saved_model_dir = PROJECT_ROOT / "saved_models"
csv_path = PROJECT_ROOT / "data" / "csv_files"

# LGE has few real slices, so the real (non-augmented) training slices are repeated per epoch.
# Augmented slices (file name contains "_aug_") are added once.
REPEAT_REAL = {"SAX_SCAR": 20, "2CH": 10, "4CH": 10, "RAS": 10, "SAX_INFARCT": 20}
CINE_CKPT_DIR = saved_model_dir / "CINE"  # CINE model used to initialise the encoder/decoder
WARMUP_EPOCHS = 100  # epochs with the encoder frozen when FREEZE_MODEL.LGE is True


def lge_views(cfg_data):
    """view name -> (csv name, config node): the CMR-MULTI LGE views plus EMIDEC as its own SAX_INFARCT head."""
    views = {key: (f"LGE_MULTI_{key}", cfg_data.CMR_MULTI.LGE_MULTI[key]) for key in cfg_data.CMR_MULTI.LGE_MULTI}
    views["SAX_INFARCT"] = ("EMIDEC_SAX_INFARCT", cfg_data.EMIDEC.SAX_INFARCT)
    return views


def load_files(cfg_data):
    """Return train/val file lists per LGE view using the csv `type` column (EMIDEC `test` rows are ignored)."""
    train_files, val_files, num_classes_dict, class_weights_dict = {}, {}, {}, {}
    for key, (csv_name, cfg_view) in lge_views(cfg_data).items():
        csv_file = pd.read_csv(csv_path / f"{csv_name}_info.csv")
        # csv paths are relative to the project root
        paths = csv_file["path"].map(lambda p: str(PROJECT_ROOT / p))  # type: ignore
        is_train, is_val = csv_file["type"] == "train", csv_file["type"] == "val"
        is_aug = csv_file["path"].str.contains("_aug_")

        real = paths[is_train & ~is_aug].tolist()
        aug = paths[is_train & is_aug].tolist()
        train_files[key] = real * REPEAT_REAL[key] + aug
        val_files[key] = paths[is_val].tolist()
        num_classes_dict[key] = cfg_view.num_classes
        class_weights_dict[key] = cfg_view.class_weights

        print(
            f"  {key}: real {len(real)} x{REPEAT_REAL[key]} + aug {len(aug)} = {len(train_files[key])} train "
            f"| {len(val_files[key])} val"
        )
    return train_files, val_files, num_classes_dict, class_weights_dict


def find_cine_checkpoint():
    """Best-dice CINE checkpoint, falling back to last.ckpt. Returns None if there is none."""
    best = sorted(glob.glob(os.path.join(CINE_CKPT_DIR, "dice_*.ckpt")))
    if best:
        return best[-1]
    last = os.path.join(CINE_CKPT_DIR, "last.ckpt")
    return last if os.path.exists(last) else None


def load_pretrained(segmenter, checkpoint):
    """Load encoder + decoder weights from a CINE checkpoint; the heads are different for LGE, so skip them."""
    state_dict = torch.load(checkpoint, map_location="cpu")["state_dict"]
    state_dict = {k: v for k, v in state_dict.items() if "finalHeads" not in k}
    segmenter.load_state_dict(state_dict, strict=False)


def find_lge_checkpoint():
    """Best-dice LGE checkpoint in saved_models/LGE (highest dice in the file name). Returns None if there is none."""
    best = sorted(glob.glob(os.path.join(saved_model_dir, "LGE", "lge_0.*.ckpt")))
    return best[-1] if best else None


def main():
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg_data = compose(config_name="data_config")
        cfg_train = compose(
            config_name="train_config",
            overrides=["BATCH_SIZE=4", "PATIENCE_LR=50", "PATIENCE_ES=300"],
        )

    print("training LGE (CMR-MULTI + EMIDEC)...")
    train_files, val_files, num_classes_dict, class_weights_dict = load_files(cfg_data)
    print(f"num_classes_dict: {num_classes_dict}")
    model = NATA_Net(in_channels=cfg_train.INPUT_DIM_MODEL, head_classes=num_classes_dict)

    segmenter = Segmenter_MultiView(
        model,
        class_weights_dict=class_weights_dict,
        num_classes_dict=num_classes_dict,
        learning_rate=cfg_train.LEARNING_RATE,
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
            name="LGE",
            resume=False,
        )
        if cfg_train.LOG_WANDB
        else False
    )
    save_dir = saved_model_dir / "LGE"
    os.makedirs(save_dir, exist_ok=True)
    # Initialize a ModelCheckpoint callback to save the model weights when the validation dice improves
    check_point = ModelCheckpoint(
        save_dir,
        filename="lge_{avg_val_dice:0.4f}",
        monitor="avg_val_dice",
        mode="max",
        save_top_k=cfg_train.SAVE_MODEL_TOP_K,
        verbose=True,
        save_weights_only=True,
        auto_insert_metric_name=False,
    )

    # Initialize a LearningRateMonitor callback to log the learning rate during training
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # Initialize a EarlyStopping callback to stop training if the validation dice does not improve for a certain number of epochs
    last_ckpt = ModelCheckpoint(
        dirpath=save_dir,
        filename="last",
        monitor=None,
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        enable_version_counter=False,
        verbose=True,
    )
    check_point_epoch = ModelCheckpoint(
        dirpath=save_dir,
        filename="lge_{epoch:03d}",
        every_n_epochs=100,
        save_top_k=-1,
        # optional: also save last.ckpt
        monitor=None,  # no metric ranking
        save_weights_only=True,
        auto_insert_metric_name=False,
    )
    # Define a dictionary with the parameters for the Trainer object
    PARAMS_TRAINER = {
        "accelerator": cfg_train.ACCELERATOR,
        "devices": cfg_train.DEVICES,
        "benchmark": True,
        "enable_progress_bar": True,
        # "overfit_batches" :5,
        "logger": wandb_logger,
        "callbacks": [check_point, lr_monitor, last_ckpt, check_point_epoch],
        "log_every_n_steps": 1,
        "num_sanity_val_steps": 5,
        "max_epochs": cfg_train.EPOCHS,
        "precision": cfg_train.PRECISION,
    }

    # If LGE weights already exist in saved_models/LGE, continue from them; otherwise initialise the
    # encoder/decoder from the CINE model
    resume_ckpt = find_lge_checkpoint()
    if cfg_train.USE_TRANSFER_LEARNING.LGE and resume_ckpt is not None:
        # weights only: the optimizer, LR schedule and epoch counter start from scratch
        print(f"found LGE weights, continue training from: {resume_ckpt}")
        segmenter.load_state_dict(torch.load(resume_ckpt, map_location="cpu")["state_dict"], strict=False)
    elif cfg_train.USE_TRANSFER_LEARNING.LGE:
        checkpoint = find_cine_checkpoint()
        if checkpoint is None:
            print(f"no CINE checkpoint found in {CINE_CKPT_DIR}, training LGE from scratch ...")
        else:
            print("using pretrained CINE model for LGE ...")
            print(f"load checkpoint: {checkpoint}")
            load_pretrained(segmenter, checkpoint)

    # Optionally train only the heads (and the first conv) for a few epochs, then unfreeze everything
    # (skipped when continuing from LGE weights: the encoder is already trained)
    if cfg_train.FREEZE_MODEL.LGE and resume_ckpt is None:
        print("freezing model weights for LGE ...")
        PARAMS_TRAINER_WARMUP = {
            "accelerator": cfg_train.ACCELERATOR,
            "devices": cfg_train.DEVICES,
            "benchmark": True,
            "enable_progress_bar": True,
            "logger": False,
            "log_every_n_steps": 1,
            "num_sanity_val_steps": 5,
            "max_epochs": WARMUP_EPOCHS,
            "precision": cfg_train.PRECISION,
        }
        segmenter.freeze_model()
        trainer = pl.Trainer(**PARAMS_TRAINER_WARMUP)
        trainer.fit(segmenter, train_loader, val_loader)
        segmenter.unfreeze_model()

    # Train the model using the train and validation data loaders
    trainer = pl.Trainer(**PARAMS_TRAINER)
    trainer.fit(segmenter, train_loader, val_loader)

    if cfg_train.LOG_WANDB:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
