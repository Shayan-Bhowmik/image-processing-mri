# Brain MRI AI Decision Support – Project Log

---

## Step 1 – Repository Initialization (Completed)

### Remote Setup
- Created GitHub repository: image-processing-mri
- Repository created without README, license, or .gitignore

### Local Setup
- Created local project directory
- Initialized Git repository using `git init`
- Created standardized folder structure:
  - src/
  - models/
  - notebooks/
  - data/
  - app/

### Repository Hygiene
- Added .gitignore to prevent tracking:
  - Dataset files (.nii, .nii.gz)
  - Model weights (.h5, .ckpt)
  - Python cache
  - Virtual environments
  - OS system files

### Version Control
- Created initial commit
- Renamed branch to `main`
- Connected local repository to GitHub remote
- Pushed initial commit to GitHub

Repository is properly initialized and synchronized.

---

## Step 2 – Python Environment Setup (Completed)

### Interpreter Configuration
- Installed Python 3.11
- Created virtual environment
- Activated isolated environment

### Deep Learning Framework
- Installed PyTorch (initially CPU version)
- Verified successful import

### Supporting Libraries Installed
- NiBabel (MRI loading)
- NumPy (numerical operations)
- Pandas (metadata handling)
- OpenCV (image preprocessing)
- Matplotlib (visualization)
- Scikit-learn (evaluation metrics)

### Reproducibility
- Generated `requirements.txt`
- Locked dependency versions

Environment configuration validated.

---

## Step 3 – Dataset Setup (Completed)

### BRATS 2020 Dataset (Abnormal Class)
- Downloaded via Kaggle API
- Extracted to: `data/raw/brats`
- Verified patient folder structure
- Confirmed availability of:
  - FLAIR modality
  - Segmentation masks

### Abnormal Label Extraction Logic
- Implemented segmentation-based labeling
- Created `src/label_utils.py`
- Verified tumor detection using BRATS segmentation mask
- Confirmed abnormal label generation (Label = 1)

### OASIS Dataset (Normal Class)
- Downloaded via Kaggle API
- Verified NIfTI volume structure
- Implemented normal label logic
- Confirmed correct label assignment (Label = 0)

Dataset layer validated.

---

## Step 4 – MRI Volume Preprocessing Pipeline (Completed)

### Step 4.1 – NIfTI Volume Loading
- Implemented loader using NiBabel
- Created: `src/preprocessing/load_nifti.py`
- Verified BRATS FLAIR volume shape: `(240, 240, 155)`
- Confirmed dtype: float32

### Step 4.2 – Z-Score Normalization
- Applied normalization only to non-zero voxels
- Prevented division-by-zero edge cases
- Verified:
  - Mean ≈ 0
  - Std ≈ 1

### Step 4.3 – Axial Slice Extraction & Filtering
- Extracted axial slices from 3D volume
- Removed near-empty slices using thresholding
- Reduced:
  - 155 slices → 126 valid slices
- Confirmed shape: `(240, 240)`

### Step 4.4 – 2.5D Slice Stacking
- Implemented neighbor-based stacking:
  `[prev, current, next]`
- Handled boundary duplication
- Output shape: `(3, 240, 240)`
- Channel-first format maintained

### Step 4.5 – Resize for CNN Compatibility
- Implemented bilinear interpolation (PyTorch)
- Resized to `(3, 224, 224)`
- Returned `torch.Tensor`

Preprocessing pipeline validated end-to-end.

---

## Step 5 – Dataset Architecture

### Step 5.1 – Leakage-Safe Patient-Level Split (Initial Version)

- Created: `src/data/split_dataset.py`
- Implemented deterministic split with fixed seed
- Enforced patient-level separation
- Persisted split to:
  - `data/splits/patient_split.json`

Initial Split Summary:
- Total patients: 350
- Train: 244
- Validation: 52
- Test: 54

⚠️ Later discovered to include only BRATS patients.

---

### Step 5.2 – Custom Slice-Level PyTorch Dataset (Initial Version)

- Created: `src/data/mri_dataset.py`
- Integrated full preprocessing pipeline inside Dataset
- Implemented slice-level index mapping
- Implemented memory-safe lazy loading
- Verified:
  - Tensor shape: `(3, 224, 224)`
  - Label output: 0/1

Initial training slice count:
- 32,405 slices

---

### Step 5.3 – Train / Validation / Test DataLoaders

- Created: `src/data/dataloaders.py`
- Connected patient-level split to slice-level dataset
- Configured:
  - Train loader (shuffle=True)
  - Validation loader (shuffle=False)
  - Test loader (shuffle=False)
- Verified batch shapes:
  - Images: `(B, 3, 224, 224)`
  - Labels: `(B)`

Data pipeline completed.

---

### Step 5.4 – Multi-Dataset Integration & Structural Refactor (Critical Fix)

#### Problem Identified
- Training dataset contained only BRATS patients
- OASIS dataset was not included
- Reported 99% accuracy was invalid (single-class bias)

#### Root Cause
- OASIS structure differs:
  - BRATS → folder-based patients
  - OASIS → direct `.nii` files
- Split logic assumed folder-only patients

#### Fix Implemented

- Rebuilt patient-level split using:
  - BRATS root:
    `data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData`
  - OASIS root:
    `data/raw/oasis/OASIS_Clean_Data/OASIS_Clean_Data`

- Updated split logic:
  - BRATS folders → label = 1
  - OASIS `.nii` files → label = 0

#### Updated Split Summary
- Total patients: 786
  - BRATS: 350
  - OASIS: 436
- Train: 550
- Validation: 118
- Test: 118

#### Dataset Class Refactor

- Removed `get_label_from_patient_folder`
- Refactored `MRIDataset` to accept split entries directly
- Added 4D volume handling:
  - Automatically squeezes extra dimension for OASIS
- Preserved volume caching
- Maintained 2.5D stacking and resizing

#### Verified Output

- Total training slices: 76,091
- Sample shape: `(3, 224, 224)`
- Labels verified
- No runtime errors
- Dataset stable

#### Architectural Impact

- Correct binary classification setup
- Eliminated single-class training bias
- Ensured statistically valid experimentation
- Established production-grade data pipeline

Dataset layer now scientifically valid and robust.

---

### Step 5.5 – Dataset Memory Stabilization & Architectural Redesign (Completed)

#### Problem Identified
- Repeated full-volume loading during `__getitem__`
- Mask-based normalization causing large temporary array allocations
- DataLoader memory crashes during slice-level iteration
- Precomputing all resized samples exceeded available RAM

#### Root Cause
- Half-lazy dataset design:
  - Volumes normalized multiple times
  - Full 3D volumes loaded repeatedly per slice
  - Mask operations created large intermediate arrays
- Parallel DataLoader workers amplified memory usage

#### Final Architectural Solution

Redesigned `MRIDataset` to:

- Load each patient volume once during initialization
- Perform normalization once per volume
- Store normalized 3D volumes in memory (volume-level caching)
- Build slice-level index map only
- Generate 2.5D slice triplets on-the-fly in `__getitem__`
- Resize samples dynamically during retrieval

#### Memory Optimization Improvements

- Updated `load_nifti()` to load directly as `float32`
- Eliminated redundant float64 allocation
- Removed repeated masked array normalization
- Prevented repeated volume reload inside DataLoader
- Disabled multiprocessing during debugging

#### Verified Stability

- Total indexed slices (train split): 76,091
- Class 0 slices: 43,615
- Class 1 slices: 32,476
- Dataset iteration completed without memory errors

#### Architectural Impact

- Dataset pipeline now memory-stable
- No repeated heavy computation
- Fully scalable to complete dataset
- Training layer ready for proper experimentation

Data infrastructure now production-grade and stable.

---

## Step 6 - Training System Hardening
### Step 6.1 — Class-Weighted Loss Implemented

File Modified:
- train.py

Changes:
- Added sklearn compute_class_weight
- Computed class weights from training data
- Updated CrossEntropyLoss to weighted version
- Verified class distribution and printed weights
- No architectural changes

---

Step 6.2 — Validation Loop Added

File Modified:
- train.py

Changes:
- Added evaluate() function
- Added validation loss and accuracy tracking
- Training now reports:
  - Train loss
  - Train accuracy
  - Validation loss
  - Validation accuracy
- Increased epochs to 5 for baseline observation
- No architecture changes

---

Step 6.2 — Validation Metrics Verified

Results (5 epochs baseline):
- Train Accuracy: 99.43%
- Validation Accuracy: 99.38%
- Train/Val curves stable
- No visible overfitting

System now reports scientifically valid training + validation metrics.

---

Step 6.3 — Validation Confusion Matrix Verified

Validation Results:
Confusion Matrix:
[[9867, 0],
 [101, 6389]]

Per-Class Metrics:
- Class 0 Recall: 100%
- Class 1 Recall: 98%
- F1-scores ~0.99 for both classes

Observations:
- Very low false negative rate (~1.5%)
- No false positives
- High stability across epochs

---

Step 6.4 — Sanity Label Shuffle Test Completed

Results:
- Train Accuracy ~56%
- Validation Accuracy ~60%
- Loss ~0.69 (random baseline)
- Model collapsed to predicting majority class

Conclusion:
- No hidden data leakage
- Split integrity verified
- Model only performs well with correct labels
- Pipeline scientifically validated

---

## Step 7 - Generalization Hardening
### Step 7.1 — Production Training Mode Enabled

File Modified:
- train.py

Changes:
- Removed sanity label shuffle
- Restored full training
- Added checkpoint directory
- Added automatic saving of best validation model
- Training now persists best_model.pth

---

Step 7.2 — Learning Rate Scheduler Added

File Modified:
- train.py

Changes:
- Added ReduceLROnPlateau scheduler
- Scheduler monitors validation loss
- Reduces LR by factor 0.5 on plateau
- Improves convergence stability