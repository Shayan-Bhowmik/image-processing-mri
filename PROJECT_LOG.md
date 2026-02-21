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

## Step 4 – MRI Volume Preprocessing Pipeline (In Progress)

### Step 4.1 – NIfTI Volume Loading (Completed)

- Implemented NIfTI loader using nibabel
- Created: src/preprocessing/load_nifti.py
- Successfully loaded BRATS FLAIR volume
- Verified shape: (240, 240, 155)
- Verified dtype: float32
- Confirmed environment and file path handling

Preprocessing pipeline initiated.