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

from segment2d import crop_resize_image, crop_resize_mask, preprocess_cmr, remap_mask
from natsort import natsorted
import csv
import re
from collections import defaultdict
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

PROJECT_ROOT = src_dir.parent

config_dir = PROJECT_ROOT / "config"
acdc_dir = PROJECT_ROOT / "data/ACDC_database/training"
save_dir = "data/ACDC_npz/SAX"
csv_dir = PROJECT_ROOT / "data/csv_files"

N_VAL_PER_GROUP = 2  # ACDC has 5 groups x 20 patients -> 90 train / 10 val
SEED = 42
# ACDC: 1 = RV, 2 = MYO, 3 = LV  ->  1 = LV, 2 = MYO, 3 = RV
LABEL_MAP = {1: 3, 2: 2, 3: 1}


def read_group(patient_dir):
    """Return the pathology group (DCM, HCM, MINF, NOR, RV) from Info.cfg."""
    with open(os.path.join(patient_dir, "Info.cfg")) as f:
        for line in f:
            if line.startswith("Group:"):
                return line.split(":")[1].strip()
    raise ValueError(f"No Group found in {patient_dir}/Info.cfg")


def split_patients(patient_dirs, n_val=N_VAL_PER_GROUP, seed=SEED):
    """Pick n_val validation patients from each pathology group; the rest are training."""
    groups = defaultdict(list)
    for patient_dir in patient_dirs:
        groups[read_group(patient_dir)].append(patient_dir)
    rng = np.random.default_rng(seed)
    val = set()
    for group in sorted(groups):
        members = groups[group]
        val.update(members[i] for i in rng.choice(len(members), size=n_val, replace=False))
    train_dirs = [p for p in patient_dirs if p not in val]
    val_dirs = [p for p in patient_dirs if p in val]
    return train_dirs, val_dirs


def process_patient(patient_dir, split, writer, resize_dim):
    id_patient = os.path.basename(patient_dir)
    # only frames that have a ground truth (ED and ES)
    for mask_path in natsorted(glob.glob(os.path.join(patient_dir, f"{id_patient}_frame*_gt.nii.gz"))):
        image_path = mask_path.replace("_gt.nii.gz", ".nii.gz")
        frame = int(re.search(r"_frame(\d+)_gt", mask_path).group(1))  # type: ignore
        image = nib.load(image_path).get_fdata()  # type: ignore
        mask = nib.load(mask_path).get_fdata().astype(np.uint8)  # type: ignore
        mask = remap_mask(mask, LABEL_MAP)  # RV -> 3, MYO -> 2, LV -> 1
        image = preprocess_cmr(image)
        resize_image, restore_info = crop_resize_image(image, resize_dim)
        resize_mask = crop_resize_mask(mask, restore_info)

        for i in range(resize_image.shape[-1]):
            if np.sum(resize_mask[:, :, i]) == 0:
                continue
            slice_image = resize_image[:, :, i]
            slice_mask = resize_mask[:, :, i]
            npz_save_path = os.path.join(save_dir, f"{id_patient}_frame{frame:02d}_{i + 1}.npz")
            np.savez_compressed(npz_save_path, image=slice_image[..., np.newaxis], mask=slice_mask.astype(np.uint8))
            writer.writerow({"id_patient": id_patient, "path": npz_save_path, "type": split})


def main():
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="data_config")

    patient_dirs = natsorted(glob.glob(str(acdc_dir / "patient*")))
    train_dirs, val_dirs = split_patients(patient_dirs)

    with open(f"{csv_dir}/ACDC_SAX_info.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["id_patient", "path", "type"])
        writer.writeheader()
        for split, dirs in (("train", train_dirs), ("val", val_dirs)):
            print(f"Processing {len(dirs)} patients for {split} set...")
            for patient_dir in tqdm(dirs):
                process_patient(patient_dir, split, writer, cfg.RESIZE_DIM)


if __name__ == "__main__":
    main()
