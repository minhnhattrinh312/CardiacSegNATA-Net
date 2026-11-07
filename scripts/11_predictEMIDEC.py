import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from medpy.metric.binary import dc
from natsort import natsorted
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
src_dir = PROJECT_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from segment2d import NATA_Net
from segment2d.utils import (
    postprocess_multiclass_volume,
    predict_patches,
    preprocess_data_nii,
    restore_mask,
    save_nii,
)

config_dir = PROJECT_ROOT / "config"
WEIGHTS = PROJECT_ROOT / "saved_models/LGE/lge_0.7279.ckpt"  # best LGE weights from scripts/07_train_LGE.py
DATA_DIR = PROJECT_ROOT / "data/emidec-dataset-1.0.1"
CSV_PATH = PROJECT_ROOT / "data/csv_files/EMIDEC_SAX_INFARCT_info.csv"  # written by scripts/06_EMIDEC2npz.py
PRED_DIR = PROJECT_ROOT / "output/EMIDEC_testset"
OUT_DIR = PROJECT_ROOT / "output"
BATCH_SIZE = 8
VIEW = "SAX_INFARCT"
# EMIDEC labels already follow the model convention: 1 = LV, 2 = normal MYO, 3 = MI, 4 = no-reflow (MVO)
# metric name -> labels that make up the structure
STRUCTURES = {
    "LV": [1],  # LV cavity
    "MYO": [2, 3, 4],  # whole myocardium
    "MI": [3, 4],  # infarction (includes no-reflow)
    "MVO": [4],  # no-reflow
}


def load_model(weight_path, num_classes, device):
    model = NATA_Net(in_channels=1, head_classes=num_classes)
    model = torch.compile(model.to(device))
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    # Lightning keys are `model._orig_mod.<name>`, i.e. already those of a torch.compile'd model
    state_dict = {k.replace("model.", "", 1): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    return model.eval()


def predict_case(model, image_path, num_classes, device):
    data = preprocess_data_nii(image_path)
    images = data["image"]
    prob = predict_patches(images, model, view=VIEW, num_classes=num_classes, batch_size=BATCH_SIZE, device=device)
    seg = np.argmax(prob, axis=1).transpose(1, 2, 0)  # (dim, dim, n_slices)
    seg = postprocess_multiclass_volume(seg, min_size=200, hole_area=50, smooth_radius=0, keep_largest=False)
    return restore_mask(seg, data["restore_info"]).astype(np.uint8), data


def scar_burden(mask):
    """Infarction (MI + no-reflow) as a percentage of the myocardium."""
    myo = np.isin(mask, STRUCTURES["MYO"]).sum()
    mi = np.isin(mask, STRUCTURES["MI"]).sum()
    return 100.0 * mi / myo if myo > 0 else np.nan


def case_metrics(pred, gt, spacing):
    voxel_vol = np.prod(spacing) / 1000.0  # mm^3 -> mL
    row = {}
    vol_gt_myo = np.isin(gt, STRUCTURES["MYO"]).sum() * voxel_vol
    for name, labels in STRUCTURES.items():
        pred_s, gt_s = np.isin(pred, labels), np.isin(gt, labels)
        # structure absent from both GT and prediction (normal cases, cases without no-reflow): perfect, Dice = 1
        row[f"dice_{name}"] = dc(pred_s, gt_s) if (gt_s.any() or pred_s.any()) else 1.0
        row[f"voldiff_{name}"] = abs(pred_s.sum() - gt_s.sum()) * voxel_vol  # mL
    for name in ("MI", "MVO"):  # as in the EMIDEC challenge: |V_pred - V_gt| relative to the GT myocardium volume
        row[f"voldiff_rate_{name}"] = 100.0 * row[f"voldiff_{name}"] / vol_gt_myo if vol_gt_myo > 0 else np.nan
    row["scar_burden_gt"] = scar_burden(gt)
    row["scar_burden_pred"] = scar_burden(pred)
    row["scar_burden_diff"] = abs(row["scar_burden_pred"] - row["scar_burden_gt"])  # percentage points
    return row


def write_markdown(df, md_path, weight_path):
    groups = {
        "All": df,
        "Normal": df[df["group"] == "N"],
        "Pathological": df[df["group"] == "P"],
    }
    lines = [
        "# EMIDEC test set results",
        "",
        f"- Weights: `{weight_path}`",
        f"- Cases: {len(df)} ({(df['group'] == 'N').sum()} normal, {(df['group'] == 'P').sum()} pathological), 3D per-volume",
        "- LV = LV cavity (label 1); MYO = normal myocardium + MI + no-reflow; MI = infarction incl. no-reflow; MVO = no-reflow",
        "- Dice in [0, 1] (higher is better); if the structure is absent in both GT and prediction Dice = 1, if absent in only one Dice = 0",
        "- Vol. diff = |V_pred - V_gt| in mL; Rate = Vol. diff / V_gt(MYO) in %",
        "- Scar burden = (MI + MVO) / MYO volume in %; Burden diff = |burden_pred - burden_gt| in percentage points",
        "- Per-case values: `emidec_testset_metrics.csv`",
        "",
    ]
    for title, sub in groups.items():
        lines += [f"## {title} ({len(sub)} cases)", ""]
        lines += ["| Structure | Dice | Vol. diff (mL) | Rate (%) |", "|---|---|---|---|"]
        for name in STRUCTURES:
            rate = f"{sub[f'voldiff_rate_{name}'].mean():.2f}" if f"voldiff_rate_{name}" in sub else "-"
            lines.append(f"| {name} | {sub[f'dice_{name}'].mean():.4f} | {sub[f'voldiff_{name}'].mean():.3f} | {rate} |")
        r = f"{sub['scar_burden_gt'].corr(sub['scar_burden_pred']):.3f}" if sub["scar_burden_gt"].std() > 0 else "-"
        lines += [
            "",
            f"Scar burden: GT {sub['scar_burden_gt'].mean():.2f}% | pred {sub['scar_burden_pred'].mean():.2f}% | "
            f"mean abs. diff {sub['scar_burden_diff'].mean():.2f} pp (Pearson r = {r})",
            "",
        ]
    md_path.write_text("\n".join(lines))


def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg_data = compose(config_name="data_config")
    # same head layout as scripts/07_train_LGE.py: the CMR-MULTI LGE views plus EMIDEC as its own SAX_INFARCT head
    num_classes = {k: v.num_classes for k, v in cfg_data.CMR_MULTI.LGE_MULTI.items()}
    num_classes[VIEW] = cfg_data.EMIDEC[VIEW].num_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(WEIGHTS, num_classes, device)

    info = pd.read_csv(CSV_PATH)
    test_cases = natsorted(info.loc[info["type"] == "test", "id_patient"].unique())

    rows = []
    for case in tqdm(test_cases):
        case_dir = DATA_DIR / case
        image_path = case_dir / "Images" / f"{case}.nii.gz"
        gt_nii = nib.load(case_dir / "Contours" / f"{case}.nii.gz")
        seg, data = predict_case(model, str(image_path), num_classes[VIEW], device)
        save_nii(str(PRED_DIR / f"{case}.nii.gz"), seg, data["affine"], data["header"])

        gt = gt_nii.get_fdata().astype(np.uint8)  # type: ignore
        spacing = gt_nii.header.get_zooms()[:3]  # type: ignore
        row = {"patient": case, "group": "N" if case.startswith("Case_N") else "P"}
        row.update(case_metrics(seg, gt, spacing))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "emidec_testset_metrics.csv", index=False, float_format="%.4f")
    write_markdown(df, OUT_DIR / "emidec_testset_results.md", WEIGHTS)
    print((OUT_DIR / "emidec_testset_results.md").read_text())


if __name__ == "__main__":
    main()
