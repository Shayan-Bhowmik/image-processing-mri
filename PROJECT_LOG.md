# Brain MRI AI Decision Support – Project Log

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

Repository is now properly initialized and synchronized.

## Step 2 – Python Environment Setup (Completed)

### Interpreter Configuration
- Installed Python 3.11
- Created virtual environment using Python 3.11
- Activated isolated environment

### Deep Learning Framework
- Installed PyTorch (CPU version)
- Verified successful import

### Supporting Libraries Installed
- NiBabel (MRI loading)
- NumPy (numerical operations)
- Pandas (metadata handling)
- OpenCV (image preprocessing)
- Matplotlib (visualization)
- Scikit-learn (evaluation metrics)

### Reproducibility
- Generated requirements.txt using pip freeze
- Locked exact dependency versions

## Step 3 – Dataset Setup (Completed)

### BRATS 2020 Dataset (Abnormal Class)
- Downloaded via Kaggle API
- Extracted to: data/raw/brats
- Verified patient folder structure
- Confirmed availability of:
  - FLAIR modality
  - Segmentation masks

### Abnormal Label Extraction Logic
- Implemented segmentation-based labeling
- Created src/label_utils.py
- Verified tumor detection using BRATS segmentation mask
- Confirmed correct abnormal label generation (Label = 1)

### OASIS Dataset (Normal Class)
- Downloaded via Kaggle API
- Verified NIfTI volume structure
- Implemented normal label logic
- Confirmed correct label assignment (Label = 0)

Dataset layer fully validated.

## Step 4 – MRI Volume Preprocessing Pipeline (Completed)

### Step 4.1 – NIfTI Volume Loading (Completed)

- Implemented NIfTI loader using nibabel
- Created: src/preprocessing/load_nifti.py
- Successfully loaded BRATS FLAIR volume
- Verified shape: (240, 240, 155)
- Verified dtype: float32
- Confirmed environment and file path handling

Preprocessing pipeline initiated.

### Step 4.2 – Z-Score Normalization (Completed)

- Implemented volume-wise Z-score normalization
- Normalization applied only to non-zero voxels
- Prevented division-by-zero edge cases
- Verified statistical correctness:
  - Mean ≈ 0
  - Std ≈ 1
- Confirmed floating-point precision within expected tolerance

MRI intensity normalization validated.

### Step 4.3 – Axial Slice Extraction & Empty Slice Filtering (Completed)

- Implemented axial slice extraction from 3D MRI volume
- Added configurable non-zero pixel threshold filtering
- Removed near-empty/background slices
- Verified slice reduction:
  - Original depth: 155 slices
  - Valid slices retained: 126
- Confirmed slice dimensions: (240, 240)

Axial slice preparation validated for downstream 2.5D stacking.

### Step 4.4 – 2.5D Slice Stacking (Completed)

- Implemented neighbor-based stacking strategy
- Constructed samples using:
  [previous_slice, current_slice, next_slice]
- Handled boundary conditions via slice duplication
- Converted 2D slices → 3-channel 2.5D samples
- Verified output shape: (3, 240, 240)
- Maintained channel-first format

Local volumetric context successfully integrated.

### Step 4.5 – Resize for CNN Compatibility (Completed)

- Implemented bilinear interpolation resizing using PyTorch
- Resized 2.5D samples from (3, 240, 240) to (3, 224, 224)
- Returned output as torch.Tensor
- Verified compatibility with pretrained CNN input requirements

CNN-ready tensor pipeline validated.

## Step 5 – Dataset Architecture (In Progress)

### Step 5.1 – Leakage-Safe Patient-Level Split (Completed)

- Created: `src/data/split_dataset.py`
- Implemented patient folder discovery utility
- Identified correct patient directory level:
  - `MICCAI_BraTS2020_TrainingData`
- Built deterministic train/val/test split function
- Used fixed random seed for reproducibility
- Enforced patient-level separation to prevent data leakage
- Persisted split to:
  - `data/splits/patient_split.json`
- Corrected `.gitignore` rule to allow tracking of `src/data/`

### Split Summary:
- Total patients: 350  
- Train: 244  
- Validation: 52  
- Test: 54  

### Architectural Impact:
- Guarantees leakage-safe experimentation
- Establishes reproducible dataset partitioning
- Prepares integration with custom PyTorch Dataset
- Forms the foundation of model training pipeline

Patient-level dataset partitioning validated and production-ready.

### Step 5.2 – Custom Slice-Level PyTorch Dataset (Completed)

- Created: `src/data/mri_dataset.py`
- Integrated preprocessing pipeline inside Dataset:
  - NIfTI loading
  - Z-score normalization
  - Axial slice filtering
  - 2.5D stacking
  - Bilinear resizing to (224, 224)
- Refactored to slice-level dataset design
- Implemented memory-safe lazy loading strategy
- Built slice index mapping to prevent full dataset memory loading
- Dataset now returns:
  - Tensor shape: (3, 224, 224)
  - Label: 0 (normal) or 1 (abnormal)

### Dataset Statistics (Training Split):
- Patients: 244
- Total slice samples: 32,405
- Output dtype: torch.float32

### Architectural Impact:
- Enables standard PyTorch DataLoader batching
- Scalable to full dataset without memory overflow
- Fully compatible with CNN training pipeline