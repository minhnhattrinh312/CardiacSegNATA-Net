import nibabel as nib
import glob
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

src_dir = Path(__file__).resolve().parents[1] / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from segment2d import crop_resize_image, crop_resize_mask, preprocess_cmr
from natsort import natsorted
import csv
from collections import defaultdict
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

PROJECT_ROOT = src_dir.parent

config_dir = PROJECT_ROOT / "config"
emidec_dir = PROJECT_ROOT / "data/emidec-dataset-1.0.1"
save_dir = "data/EMIDEC_npz/SAX_INFARCT"
csv_dir = PROJECT_ROOT / "data/csv_files"

VAL_RATIO = 0.1
TEST_RATIO = 0.2  # train gets the rest (0.7)
SEED = 42
# EMIDEC labels already follow the NATA-Net convention: 1 = LV, 2 = normal MYO, 3 = MI, 4 = no-reflow
NO_REFLOW = 4


def mask_path_of(case_dir):
    name = os.path.basename(case_dir)
    return os.path.join(case_dir, "Contours", f"{name}.nii.gz")


def image_path_of(case_dir):
    name = os.path.basename(case_dir)
    return os.path.join(case_dir, "Images", f"{name}.nii.gz")


def stratum_of(case_dir):
    """Normal (N) / pathological without no-reflow (P) / pathological with no-reflow (P_MO)."""
    if os.path.basename(case_dir).startswith("Case_N"):
        return "N"
    mask = np.asarray(nib.load(mask_path_of(case_dir)).dataobj)  # type: ignore
    return "P_MO" if np.any(mask == NO_REFLOW) else "P"


def split_cases(case_dirs, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=SEED):
    """Split patients into train/val/test with the same ratio inside every stratum."""
    strata = defaultdict(list)
    for case_dir in case_dirs:
        strata[stratum_of(case_dir)].append(case_dir)
    rng = np.random.default_rng(seed)
    split_of = {}
    for stratum in sorted(strata):
        members = strata[stratum]
        order = rng.permutation(len(members))
        n_test = round(test_ratio * len(members))
        n_val = round(val_ratio * len(members))
        for rank, idx in enumerate(order):
            split_of[members[idx]] = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"
    return {s: [c for c in case_dirs if split_of[c] == s] for s in ("train", "val", "test")}, strata


def process_case(case_dir, split, writer, resize_dim):
    id_patient = os.path.basename(case_dir)
    image = nib.load(image_path_of(case_dir)).get_fdata()  # type: ignore
    mask = nib.load(mask_path_of(case_dir)).get_fdata().astype(np.uint8)  # type: ignore
    image = preprocess_cmr(image)
    resize_image, restore_info = crop_resize_image(image, resize_dim)
    resize_mask = crop_resize_mask(mask, restore_info)

    for i in range(resize_image.shape[-1]):
        if np.sum(resize_mask[:, :, i]) == 0:
            continue
        slice_image = resize_image[:, :, i]
        slice_mask = resize_mask[:, :, i]
        npz_save_path = os.path.join(save_dir, f"{id_patient}_{i + 1}.npz")
        np.savez_compressed(npz_save_path, image=slice_image[..., np.newaxis], mask=slice_mask.astype(np.uint8))
        writer.writerow({"id_patient": id_patient, "path": npz_save_path, "type": split})


def main():
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="data_config")

    case_dirs = natsorted(glob.glob(str(emidec_dir / "Case_*/")))
    case_dirs = [c.rstrip("/") for c in case_dirs]
    splits, strata = split_cases(case_dirs)
    for stratum, members in sorted(strata.items()):
        counts = {s: sum(c in splits[s] for c in members) for s in splits}
        print(f"{stratum}: {len(members)} cases -> {counts}")

    with open(f"{csv_dir}/EMIDEC_SAX_INFARCT_info.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["id_patient", "path", "type"])
        writer.writeheader()
        for split, dirs in splits.items():
            print(f"Processing {len(dirs)} patients for {split} set...")
            for case_dir in tqdm(dirs):
                process_case(case_dir, split, writer, cfg.RESIZE_DIM)


if __name__ == "__main__":
    main()
