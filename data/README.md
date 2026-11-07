# Data

Everything in `data/` is git-ignored except this file (see `.gitignore`). Raw datasets are downloaded by hand into
this folder; the preprocessing scripts turn them into 2D slices (`.npz`) plus a CSV index that training reads.

## NATA-Net label convention

NATA-Net has one segmentation head per view. Label `0` is always background. The label values below are what the
model sees **after preprocessing**, so a raw dataset with a different convention is remapped by its script.

### CINE

| View | 1 | 2 | 3 | 4 | 5 |
|------|---|---|---|---|---|
| SAX  | Left ventricle cavity (LV) | Myocardium (MYO) | Right ventricle cavity (RV) | | |
| 2CH  | Left ventricle cavity | Left ventricle myocardium | | | |
| 4CH  | Left ventricle cavity | Left ventricle myocardium | Right ventricle cavity | Right atrium | Left atrium |

CINE SAX is trained on three datasets that all end up with the same labels:

| Dataset | Raw labels | Remap in script |
|---------|------------|-----------------|
| CMR-MULTI | 1 = MYO, 2 = LV, 3 = RV | `{1: 2, 2: 1, 3: 3}` (`01_CMR_MULTI2npz.py`) |
| ACDC | 1 = RV, 2 = MYO, 3 = LV | `{1: 3, 2: 2, 3: 1}` (`02_ACDC2npz.py`) |
| M&Ms | 1 = LV, 2 = MYO, 3 = RV | none (`03_MnM2npz.py`) |

### LGE

| View | 1 | 2 | 3 | 4 |
|------|---|---|---|---|
| SAX_SCAR   | LV | MYO | Scar | RV |
| SAX_INFARCT | LV | Normal MYO | MI (infarct) | No-reflow |
| 2CH        | Left ventricle cavity | Left ventricle myocardium | Myocardial scar | |
| 4CH        | Left ventricle cavity | Left ventricle myocardium | Myocardial scar | Right ventricle cavity |
| RAS        | Right atrium | | | |

`SAX_INFARCT` comes from EMIDEC (`06_EMIDEC2npz.py`). EMIDEC already uses this convention, so no remap is needed. It is trained as its own
`SAX_INFARCT` head (5 classes) together with the CMR-MULTI LGE views in `scripts/07_train_LGE.py`.

The number of classes per head (including background) and the class weights live in `config/data_config.yaml`.

## Directory tree

`data/` as it looks after downloading the raw data and running scripts 01-03 and 06 (abbreviated with `...`):

```text
data/
├── README.md                     # this file (the only tracked file in data/)
│
├── CMR-MULTI/                    # raw: CMR-MULTI challenge, 3D .nii.gz
│   ├── CINE_MULTI/
│   │   ├── SAX_TR/{image,anno}/  # one .nii.gz per patient, same filename in image/ and anno/
│   │   ├── SAX_VAL/...           # validation (4CH and 2CH have the same _TR / _VAL layout)
│   │   ├── 2CH_TR/ 2CH_VAL/ 4CH_TR/ 4CH_VAL/
│   │   └── *.xlsx, *.json        # dataset metadata shipped with the challenge
│   └── LGE_MULTI/
│       └── SAX_TR/ SAX_VAL/ 2CH_TR/ 2CH_VAL/ 4CH_TR/ 4CH_VAL/ RAS_TR/ RAS_VAL/
│
├── ACDC_database/                # raw: ACDC
│   ├── training/patientXXX/      # Info.cfg, patientXXX_4d.nii.gz,
│   │                             # patientXXX_frameYY.nii.gz and patientXXX_frameYY_gt.nii.gz (ED and ES)
│   └── testing/
│
├── MnM/                          # raw: M&Ms, 4D (H, W, Z, T) .nii.gz
│   ├── Training/{Labeled,Unlabeled}/<id>/   # <id>_sa.nii.gz and <id>_sa_gt.nii.gz
│   ├── Validation/<id>/
│   ├── Testing/<id>/
│   └── 211230_M&Ms_Dataset_information_diagnosis_opendataset.csv
│
├── emidec-dataset-1.0.1/         # raw: EMIDEC, 3D LGE .nii.gz (100 cases: Case_N* normal, Case_P* pathological)
│   ├── Case_P072/{Images,Contours}/Case_P072.nii.gz
│   └── Case P072.txt             # clinical information
│
├── MnM2/  "MyoPS 2020 Dataset"/  # raw, not used by the current scripts
│
├── CMR-MULTI_npz/                # preprocessed by 01_CMR_MULTI2npz.py
│   ├── CINE_MULTI/{SAX,2CH,4CH}/ # <patient>_<slice>.npz
│   └── LGE_MULTI/{SAX_SCAR,2CH,4CH,RAS}/   # also <patient>_<slice>_aug_<j>.npz (scar augmentation)
├── ACDC_npz/SAX/                 # preprocessed by 02_ACDC2npz.py: patientXXX_frameYY_<slice>.npz
├── MnM_npz/SAX/                  # preprocessed by 03_MnM2npz.py:  <id>_t<frame>_<slice>.npz
├── EMIDEC_npz/SAX_INFARCT/       # preprocessed by 06_EMIDEC2npz.py: Case_XXXX_<slice>.npz
│
└── csv_files/                    # index of every .npz and whether it is train, val (or test, EMIDEC only)
    ├── CINE_MULTI_{SAX,2CH,4CH}_info.csv
    ├── LGE_MULTI_{SAX_SCAR,2CH,4CH,RAS}_info.csv
    ├── ACDC_SAX_info.csv
    ├── MnM_SAX_info.csv
    └── EMIDEC_SAX_INFARCT_info.csv
```

Outside `data/`, training writes checkpoints to `saved_models/` (also git-ignored).

## Managing the data folder

**1. Put raw data in place.** Download each dataset into the folder shown above and keep its original structure;
the scripts glob for these exact paths. Do not edit or rename raw files.

**2. Preprocess.** Run from the project root (`save_dir` is relative to the current directory):

```bash
python scripts/01_CMR_MULTI2npz.py   # CMR-MULTI -> data/CMR-MULTI_npz
python scripts/02_ACDC2npz.py        # ACDC      -> data/ACDC_npz
python scripts/03_MnM2npz.py         # M&Ms      -> data/MnM_npz
python scripts/06_EMIDEC2npz.py      # EMIDEC    -> data/EMIDEC_npz
```

Each script rewrites its CSV(s) in `data/csv_files/` on every run. Delete the matching `*_npz` folder first if you
change preprocessing, otherwise stale slices stay on disk (they are no longer indexed, but they take space).

**3. What preprocessing does.**
- Intensity preprocessing (`preprocess_cmr`), crop to the heart region and resize to `RESIZE_DIM` (256).
- Splits volumes into 2D slices and **skips slices with an empty mask**.
- 4D data (M&Ms) is reduced to the time frames that have a label.
- Labels are remapped to the NATA-Net convention above.

**4. `.npz` format.** Each file holds `image` with shape `(256, 256, 1)` (float32, scaled to [0, 1]) and `mask` with
shape `(256, 256)` (uint8, labels as above).

**5. CSV format.** Columns `id_patient,path,type`. `type` is `train` or `val` (EMIDEC also has `test`). `path` is relative to the project root.
Splits are by patient, never by slice, so slices of one patient never appear in both sets.

| Dataset | Split |
|---------|-------|
| CMR-MULTI | official `_TR` -> train, `_VAL` -> val |
| ACDC | `training/` only: 90 train / 10 val, 2 validation patients per pathology group (DCM, HCM, MINF, NOR, RV), fixed seed 42 |
| M&Ms | `Training/Labeled` + `Training/Unlabeled` -> train, `Validation` -> val |
| EMIDEC | all 100 cases: 70 train / 10 val / 20 test, fixed seed 42. The same 70/10/20 ratio is applied separately to three groups: normal (33), pathological without no-reflow (27, no label 4) and pathological with no-reflow (40) |

**6. Training.** `scripts/04_train_CINE.py` reads the CSVs, merges ACDC and M&Ms into the CINE SAX view and trains
with a single train/val split (no folds).

**7. Adding a dataset.** Write a `scripts/NN_<name>2npz.py` that (a) remaps labels to the convention above,
(b) writes slices as `.npz` under `data/<name>_npz/<view>/`, and (c) writes `data/csv_files/<name>_<view>_info.csv`
with `id_patient,path,type`. Then add the CSV name to `EXTRA_CSV` in `scripts/04_train_CINE.py`.
