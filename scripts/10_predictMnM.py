import glob
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from medpy.metric.binary import dc, hd95
from natsort import natsorted
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
src_dir = PROJECT_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from segment2d import NATA_Net
from segment2d.utils import (
    crop_resize_image,
    postprocess_multiclass_volume,
    predict_patches,
    preprocess_cmr,
    restore_mask,
    save_nii,
)

config_dir = PROJECT_ROOT / "config"
WEIGHTS = PROJECT_ROOT / "saved_models" / "CINE" / "last.ckpt"  # best CINE weights from scripts/04_train_CINE.py
DATA_DIR = PROJECT_ROOT / "data" / "MnM" / "Testing"  # <id>/<id>_sa.nii.gz (4D) + <id>_sa_gt.nii.gz
INFO_CSV = PROJECT_ROOT / "data" / "MnM" / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"  # ED/ES frames
PRED_DIR = PROJECT_ROOT / "output" / "MnM_testset"
OUT_DIR = PROJECT_ROOT / "output"
BATCH_SIZE = 8
# M&Ms labels already follow the model convention (1 = LV, 2 = MYO, 3 = RV), so no remapping is needed
CLASSES = {"LV": 1, "MYO": 2, "RV": 3}  # model label convention
GROUPS = {
    "mean(LV,MYO,RV)": ["LV", "MYO", "RV"],
}


def safe_hd95(pred, gt, spacing):
    """medpy raises when either mask is empty; return NaN instead (skipped in the means)."""
    try:
        return hd95(pred, gt, voxelspacing=spacing)
    except RuntimeError:
        return np.nan


def load_model(weight_path, num_classes, device):
    model = NATA_Net(in_channels=1, head_classes=num_classes)
    model = torch.compile(model.to(device))
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    # Lightning keys are `model._orig_mod.<name>`, i.e. already those of a torch.compile'd model
    state_dict = {k.replace("model.", "", 1): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    return model.eval()


def predict_frame(model, volume, num_classes, device):
    """volume: one 3D frame (H, W, n_slices). Same preprocessing as `preprocess_data_nii`, on an in-memory frame."""
    image = preprocess_cmr(volume)
    resized_image, restore_info = crop_resize_image(image, 256)
    images = torch.from_numpy(resized_image.transpose(2, 0, 1)[:, None]).float()  # (n_slices, 1, 256, 256)
    prob = predict_patches(images, model, view="SAX", num_classes=num_classes, batch_size=BATCH_SIZE, device=device)
    seg = np.argmax(prob, axis=1).transpose(1, 2, 0)  # (dim, dim, n_slices)
    seg = postprocess_multiclass_volume(seg, min_size=200, hole_area=0, smooth_radius=0, keep_largest=False)
    return restore_mask(seg, restore_info).astype(np.uint8)


def write_markdown(df, md_path, weight_path, n_patients):
    lines = [
        "# M&Ms test set results",
        "",
        f"- Weights: `{weight_path}`",
        f"- Patients: {n_patients} | Frames: {len(df)} (ED + ES)",
        "- Dice in [0, 1] (higher is better), HD95 in mm and volume difference |V_pred - V_gt| in mL (lower is better); 3D per-volume, mean over all cases (ED and ES pooled).",
        "- Metrics from `medpy.metric.binary` (`dc`, `hd95`).",
        "- Per-case values: `mnm_testset_metrics.csv`",
        "",
    ]
    lines += ["| Class | Dice | HD95 (mm) | Vol. diff (mL) |", "|---|---|---|---|"]
    for name in CLASSES:
        lines.append(
            f"| {name} | {df[f'dice_{name}'].mean():.4f} | {df[f'hd95_{name}'].mean():.3f} | {df[f'voldiff_{name}'].mean():.3f} |"
        )
    for group, members in GROUPS.items():
        dice = df[[f"dice_{m}" for m in members]].mean(axis=1).mean()
        hd = df[[f"hd95_{m}" for m in members]].mean(axis=1).mean()
        vol = df[[f"voldiff_{m}" for m in members]].mean(axis=1).mean()
        lines.append(f"| **{group}** | **{dice:.4f}** | **{hd:.3f}** | **{vol:.3f}** |")
    lines.append("")
    md_path.write_text("\n".join(lines))


def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg_data = compose(config_name="data_config")
    num_classes = {k: v.num_classes for k, v in cfg_data.CMR_MULTI.CINE_MULTI.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(WEIGHTS, num_classes, device)

    info = pd.read_csv(INFO_CSV, index_col="External code")
    image_paths = natsorted(glob.glob(os.path.join(DATA_DIR, "*", "*_sa.nii.gz")))
    rows = []
    for image_path in tqdm(image_paths):
        patient = os.path.basename(image_path).replace("_sa.nii.gz", "")
        image_nii = nib.load(image_path)
        image = image_nii.get_fdata()  # type: ignore  # (H, W, Z, T)
        gt = nib.load(image_path.replace("_sa.nii.gz", "_sa_gt.nii.gz")).get_fdata().astype(np.uint8)  # type: ignore
        spacing = image_nii.header.get_zooms()[:3]  # type: ignore
        voxel_vol = np.prod(spacing) / 1000.0  # mm^3 -> mL

        # only ED and ES carry a label; their frame indices come from the dataset information csv
        for phase, t in {"ED": int(info.loc[patient, "ED"]), "ES": int(info.loc[patient, "ES"])}.items():
            seg = predict_frame(model, image[..., t], num_classes["SAX"], device)

            # the spatial affine is shared by all frames; the header is the 4D one reduced to this 3D frame
            header3d = image_nii.header.copy()  # type: ignore
            header3d.set_data_shape(seg.shape)
            header3d.set_zooms(spacing)
            save_nii(str(PRED_DIR / f"{patient}_frame{t:02d}.nii.gz"), seg, image_nii.affine, header3d)  # type: ignore

            gt_t = gt[..., t]
            row = {"patient": patient, "frame": t, "phase": phase}
            for name, label in CLASSES.items():
                pred_c, gt_c = seg == label, gt_t == label
                row[f"dice_{name}"] = dc(pred_c, gt_c)
                row[f"hd95_{name}"] = safe_hd95(pred_c, gt_c, spacing)
                row[f"voldiff_{name}"] = abs(pred_c.sum() * voxel_vol - gt_c.sum() * voxel_vol)  # mL
            row["dice_mean"] = np.nanmean([row[f"dice_{n}"] for n in CLASSES])
            row["hd95_mean"] = np.nanmean([row[f"hd95_{n}"] for n in CLASSES])
            row["voldiff_mean"] = np.nanmean([row[f"voldiff_{n}"] for n in CLASSES])
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "mnm_testset_metrics.csv", index=False, float_format="%.4f")
    write_markdown(df, OUT_DIR / "mnm_testset_results.md", WEIGHTS, len(image_paths))
    print((OUT_DIR / "mnm_testset_results.md").read_text())


if __name__ == "__main__":
    main()
