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
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

PROJECT_ROOT = src_dir.parent

config_dir = PROJECT_ROOT / "config"
mnm_dir = PROJECT_ROOT / "data/MnM"
save_dir = "data/MnM_npz/SAX"
csv_dir = PROJECT_ROOT / "data/csv_files"

# M&Ms labels are already 1 = LV, 2 = MYO, 3 = RV, so no remapping is needed.
# (split name, type written to csv, folders under data/MnM)
SPLITS = [
    ("train", ["Training/Labeled", "Training/Unlabeled"]),
    ("val", ["Validation"]),
]


def process_patient(image_path, split, writer, resize_dim):
    id_patient = os.path.basename(image_path).replace("_sa.nii.gz", "")
    mask_path = image_path.replace("_sa.nii.gz", "_sa_gt.nii.gz")
    image = nib.load(image_path).get_fdata()  # type: ignore  # (H, W, Z, T)
    mask = nib.load(mask_path).get_fdata().astype(np.uint8)  # type: ignore

    # find which time frames have a label
    t_frames = [t for t in range(mask.shape[-1]) if np.sum(mask[..., t]) > 0]

    for t in t_frames:
        image_t = preprocess_cmr(image[..., t])
        resize_image, restore_info = crop_resize_image(image_t, resize_dim)
        resize_mask = crop_resize_mask(mask[..., t], restore_info)

        for i in range(resize_image.shape[-1]):
            if np.sum(resize_mask[:, :, i]) == 0:
                continue
            slice_image = resize_image[:, :, i]
            slice_mask = resize_mask[:, :, i]
            npz_save_path = os.path.join(save_dir, f"{id_patient}_t{t:02d}_{i + 1}.npz")
            np.savez_compressed(npz_save_path, image=slice_image[..., np.newaxis], mask=slice_mask.astype(np.uint8))
            writer.writerow({"id_patient": id_patient, "path": npz_save_path, "type": split})


def main():
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="data_config")

    with open(f"{csv_dir}/MnM_SAX_info.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["id_patient", "path", "type"])
        writer.writeheader()
        for split, folders in SPLITS:
            list_patient = []
            for folder in folders:
                list_patient += natsorted(glob.glob(str(mnm_dir / folder / "*/*_sa.nii.gz")))
            print(f"Processing {len(list_patient)} patients for {split} set...")
            for image_path in tqdm(list_patient):
                process_patient(image_path, split, writer, cfg.RESIZE_DIM)


if __name__ == "__main__":
    main()
