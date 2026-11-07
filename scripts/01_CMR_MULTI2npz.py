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

from segment2d import crop_resize_image, crop_resize_mask, augment_scar_only_elastic, preprocess_cmr, remap_mask
from natsort import natsorted
import csv
from tqdm import tqdm
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

PROJECT_ROOT = src_dir.parent

config_dir = PROJECT_ROOT / "config"
save_dir = "data/CMR-MULTI_npz"
csv_dir = PROJECT_ROOT / "data/csv_files"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    cfg = compose(config_name="data_config")

# loop through all folder in data config
for cmr_type in cfg.CMR_MULTI.keys():
    os.makedirs(os.path.join(save_dir, cmr_type), exist_ok=True)
    print(f"Processing {cmr_type}...")
    for cmr_view in cfg.CMR_MULTI[cmr_type].keys():
        print(f"Processing {cmr_view}...")
        os.makedirs(os.path.join(save_dir, cmr_type, cmr_view), exist_ok=True)
        path = PROJECT_ROOT / cfg.CMR_MULTI[cmr_type][cmr_view].path
        path_val = PROJECT_ROOT / cfg.CMR_MULTI[cmr_type][cmr_view].path_val
        list_patient = natsorted(glob.glob(os.path.join(path, "image/*")))
        list_patient_val = natsorted(glob.glob(os.path.join(path_val, "image/*")))
        # write to csv file for training set
        with open(f"{csv_dir}/{cmr_type}_{cmr_view}_info.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=["id_patient", "path", "type"])
            writer.writeheader()
            print(f"Processing {len(list_patient)} patients for training set...")
            for image_path in tqdm(list_patient):
                id_patient = image_path.split("/")[-1].split(".")[0]
                anno_path = image_path.replace("image", "anno")
                slice_count = 0
                image = nib.load(image_path).get_fdata()  # type: ignore
                mask = nib.load(anno_path).get_fdata().astype(np.uint8)  # type: ignore
                # remap for cine sax
                if cmr_type == "CINE_MULTI" and cmr_view == "SAX":
                    mask = remap_mask(mask, {1: 2, 2: 1, 3: 3})  # MYO -> 2, LV -> 1, scar -> 3
                image = preprocess_cmr(image)
                resize_image, restore_info = crop_resize_image(image, cfg.RESIZE_DIM)
                resize_mask = crop_resize_mask(mask, restore_info)

                for i in range(resize_image.shape[-1]):
                    if np.sum(resize_mask[:, :, i]) == 0:
                        continue
                    slice_image = resize_image[:, :, i]
                    slice_mask = resize_mask[:, :, i]
                    slice_count += 1
                    npz_save_path = os.path.join(save_dir, cmr_type, cmr_view, f"{id_patient}_{slice_count}.npz")
                    np.savez_compressed(
                        npz_save_path, image=slice_image[..., np.newaxis], mask=slice_mask.astype(np.uint8)
                    )
                    writer.writerow({"id_patient": id_patient, "path": npz_save_path, "type": "train"})
                    if cmr_type == "LGE_MULTI":
                        # augment the slice_image and slice_mask if it has scar label
                        if np.any(slice_mask == 3) and np.any(slice_mask == 2):
                            for j in range(10):  # augment 10 times
                                slice_image_aug, slice_mask_aug = augment_scar_only_elastic(slice_image, slice_mask)
                                npz_save_path_aug = os.path.join(
                                    save_dir, cmr_type, cmr_view, f"{id_patient}_{slice_count}_aug_{j+1}.npz"
                                )
                                np.savez_compressed(
                                    npz_save_path_aug,
                                    image=slice_image_aug[..., np.newaxis],
                                    mask=slice_mask_aug.astype(np.uint8),
                                )
                                writer.writerow(
                                    {"id_patient": id_patient, "path": npz_save_path_aug, "type": "train"}
                                )
            # validation set
            print(f"Processing {len(list_patient_val)} patients for validation set...")
            for image_path in tqdm(list_patient_val):
                id_patient = image_path.split("/")[-1].split(".")[0]
                anno_path = image_path.replace("image", "anno")
                slice_count = 0
                image = nib.load(image_path).get_fdata() # type: ignore
                mask = nib.load(anno_path).get_fdata().astype(np.uint8) # type: ignore
                # remap for cine sax
                if cmr_type == "CINE_MULTI" and cmr_view == "SAX":
                    mask = remap_mask(mask, {1: 2, 2: 1, 3: 3})  # MYO -> 2, LV -> 1, scar -> 3
                image = preprocess_cmr(image)
                resize_image, restore_info = crop_resize_image(image, cfg.RESIZE_DIM)
                resize_mask = crop_resize_mask(mask, restore_info)

                for i in range(resize_image.shape[-1]):
                    if np.sum(resize_mask[:, :, i]) == 0:
                        continue
                    slice_image = resize_image[:, :, i]
                    slice_mask = resize_mask[:, :, i]
                    slice_count += 1
                    npz_save_path = os.path.join(save_dir, cmr_type, cmr_view, f"{id_patient}_{slice_count}.npz")
                    np.savez_compressed(
                        npz_save_path, image=slice_image[..., np.newaxis], mask=slice_mask.astype(np.uint8)
                    )
                    writer.writerow({"id_patient": id_patient, "path": npz_save_path, "type": "val"})
