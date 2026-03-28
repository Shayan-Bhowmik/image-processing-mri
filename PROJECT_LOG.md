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

---

Step 7.3 – Data Pipeline Cleanup (Part 1)

- Fixed tensor reconstruction warning in resize.py
- Removed deprecated verbose from LR scheduler
- Replaced torch.tensor() with torch.from_numpy() in dataset
- Eliminated unnecessary tensor copies
- Verified clean training run (no warnings)

---

Step 7.3 – Data Pipeline Finalization (Part 2)

- Fixed tensor reconstruction in resize.py
- Removed deprecated verbose scheduler parameter
- Replaced torch.tensor() with torch.from_numpy() in dataset
- Added deterministic seed control
- Verified stable reproducible training
- Confirmed clean validation metrics

---

## Step 8 - True Generalization Testing

Step 8.1 – True Test Set Evaluation

- Added strict held-out test evaluation using best checkpoint
- Reported test confusion matrix and classification report
- Verified no generalization gap
- Sensitivity ≈ 98.8%
- Specificity = 100%
- Fixed torch.load future warning (weights_only=True)

---

Step 8.2 – ROC-AUC Evaluation

- Added probability extraction
- Implemented ROC curve computation
- Achieved ROC-AUC = 0.9998
- Confirmed strong separability between classes

---

Step 8.3 – Patient ID Integration

- Modified MRIDataset to return patient_id
- Updated training loop to handle 3-tuple batches
- Updated evaluate() to track patient IDs
- Verified slice-level metrics unchanged
- Confirmed stable ROC-AUC (0.9998)

---

Step 8.4 – Patient-Level Aggregation (Max Strategy)

- Implemented slice-to-patient probability aggregation
- Used max probability strategy
- Achieved 100% patient-level accuracy
- Patient-level ROC-AUC = 1.0000
- Zero false negatives at patient level

--- 

Step 8.5 – Patient-Level Threshold Optimization

- Implemented threshold sweep using Youden Index
- Optimal threshold identified: 0.2312
- Sensitivity = 1.0
- Specificity = 1.0
- Confirmed full separability at patient level
- System now supports slice-level and patient-level evaluation

---

## Step 9 Aggregation Strategy Comparison

- Implemented Max, Mean, and Fraction aggregation
- Compared ROC-AUC across strategies
- All strategies achieved perfect patient-level ROC-AUC (1.0000)
- Confirmed robustness of aggregation rule

---

## Step 10 — Explainability (Grad-CAM)

Implemented Grad-CAM explainability to visualize which brain regions the CNN uses to make tumor predictions.

Purpose:
Provide interpretable visual explanations showing model attention regions on MRI slices.

Files Added:
- src/utils/gradcam.py
- visualize_gradcam.py

Grad-CAM allows inspection of CNN decision regions and improves transparency for medical AI applications.

---

### Step 10.1 — Grad-CAM Module Implementation

Created Grad-CAM implementation for convolutional feature maps.

File:
src/utils/gradcam.py

Key Components Implemented:
- Forward hooks capture feature map activations
- Backward hooks capture gradients
- Channel importance weighting using global average pooling
- Heatmap generation using ReLU activation
- Bilinear upsampling to match MRI resolution

Output:
Heatmap highlighting CNN attention regions responsible for tumor predictions.

---

### Step 10.2 — Grad-CAM Visualization Script

Created visualization script to run Grad-CAM on MRI slices.

File:
visualize_gradcam.py

Functionality:
- Loads trained CNN model
- Loads test dataset
- Selects MRI slices
- Generates Grad-CAM heatmaps
- Displays:
  - MRI slice
  - Grad-CAM heatmap
  - Overlay visualization

This allows qualitative inspection of model behavior.

---

### Step 10.3 — Batch Grad-CAM Generation

Extended visualization to automatically generate multiple Grad-CAM samples.

Changes:
- Iterate through test dataset
- Skip near-empty MRI slices
- Automatically generate Grad-CAM visualizations
- Save output images

Output Directory:
results/gradcam/

Example Output Files:
gradcam_1.png  
gradcam_2.png  
gradcam_3.png  
gradcam_4.png  
gradcam_5.png  

Purpose:
Produce explainability artifacts suitable for:
- research reports
- project demonstrations
- technical presentations
- exhibitions

---

### Step 10.4 — Grad-CAM Visualization Improvements

Observed Issue:

Initial Grad-CAM outputs highlighted background noise instead of clear brain regions.

Root Causes:
- MRI slices contain large background regions
- CNN attention maps extended outside the brain area
- Direct overlay reduced visual clarity

Fixes Implemented:

1. Brain Region Masking  
Applied intensity thresholding to isolate brain tissue and suppress background activation.

2. Heatmap Normalization  
Grad-CAM heatmaps were re-normalized after masking to improve visibility.

3. Improved Overlay Visualization  
Generated three-panel output for better interpretability:
- MRI slice
- Grad-CAM heatmap
- MRI + Grad-CAM overlay

Result:

Grad-CAM outputs now highlight tumor-related regions within the brain more clearly while reducing background noise.

This significantly improves explainability quality for demonstrations and evaluation.

Grad-CAM pipeline successfully integrated into the project.

---

## Step 10.5 — Grad-CAM Debugging & Stabilization (Critical Fix)

### Problem Identified

Grad-CAM outputs became unstable:

- Fully blue heatmaps (no activation)
- Over-saturated maps
- Noisy visualizations

---

### Root Causes

1. Incorrect target layer (`features[-1]`) → no spatial gradients  
2. Over-aggressive thresholding → removed valid signal  
3. Cropping/padding → distorted CAM  
4. Weak gradients → required full signal preservation  

---

### Fixes Implemented

#### Correct Layer Selection
```python```
target_layer = model.features[6]

---

## Step 11 — Ablation Framework Setup

Implemented configurable training pipeline to support ablation experiments.

### Changes Made

- Added `use_2_5d` flag in MRIDataset
  - Supports both:
    - 2.5D input (3 slices)
    - Single-slice input (1 channel)

- Updated dataloaders to pass configuration flag
  - `create_dataloaders(..., use_2_5d=True/False)`

- Modified CNN model to support dynamic input channels
  - Added `in_channels` parameter
  - Enables switching between 1-channel and 3-channel input

- Updated training pipeline (`train.py`)
  - Introduced config-based control:
    - `use_2_5d`
  - Model and dataloaders now adapt automatically

### Verification

- Ran full training with `use_2_5d=True`
- Achieved same baseline performance:
  - Test Accuracy: ~99.47%
  - ROC-AUC: ~0.9998
  - Patient-level accuracy: 100%

### Outcome

System is now fully configurable for controlled ablation experiments without code duplication.

Ready to evaluate impact of architectural choices.

--- 

## Step 11.1 — Ablation 1: Removal of 2.5D Context

### Objective

Evaluate the impact of spatial context (2.5D input) by removing neighboring slices and using only single-slice (2D) input.

---

### Changes Implemented

#### Data Pipeline Modification

File Modified:
- `src/preprocessing/stacking.py`

Changes:
- Removed multi-slice stacking:
  - `[prev_slice, current_slice, next_slice]`
- Replaced with single-slice input:
  - `[current_slice]`
- Added channel dimension using `np.expand_dims`
- Output shape updated:
  - From `(3, H, W)` → `(1, H, W)`

---

### Rationale

- 2.5D input captures inter-slice spatial continuity
- This experiment isolates the contribution of contextual information
- Establishes whether neighboring slices significantly impact classification performance

---

### Experimental Integrity

- No changes made to:
  - Model architecture (except input channels – pending)
  - Loss function
  - Optimizer
  - Dataset split
  - Training schedule

- Ensures controlled ablation with only one variable changed

---

### Status

- Data pipeline successfully modified
- Model update pending
- Training not yet started

---

## Step 11.2 — Ablation 1 Results (2D vs 2.5D)

### Results Summary

| Model Variant | Input Type | Test Accuracy | ROC-AUC |
|--------------|----------|--------------|--------|
| Baseline     | 2.5D     | 99.47%       | 0.9998 |
| Ablation 1   | 2D       | 99.32%       | 0.9996 |

### Observations

- Slight decrease in accuracy (~0.15%)
- Slight drop in ROC-AUC
- Minor increase in false negatives
- No increase in false positives

### Interpretation

- 2.5D context provides additional spatial information across slices
- However, tumor features are sufficiently strong in individual slices
- Model remains highly effective even without contextual input

### Conclusion

2.5D input improves performance marginally, particularly in reducing false negatives, but is not strictly necessary for high classification accuracy.

---

## Step 11.3 — Ablation 2: Removal of Class Weights

### Objective

Evaluate the impact of class imbalance handling by removing class weights from the loss function.

---

### Changes Implemented

- Removed class weights from CrossEntropyLoss
- Replaced:
  - `CrossEntropyLoss(weight=class_weights)`
  - with `CrossEntropyLoss()`

---

### Results Summary

| Model Variant | Class Weights | Test Accuracy | ROC-AUC |
|--------------|--------------|--------------|--------|
| Baseline     | Yes          | 99.47%       | 0.9998 |
| Ablation 2   | No           | 99.41%       | 0.9998 |

---

### Observations

- Negligible drop in accuracy (~0.06%)
- ROC-AUC unchanged
- Slight reduction in false negatives
- No bias toward majority class observed

---

### Interpretation

- Dataset imbalance is mild (~1.34:1 ratio)
- Model learns class distribution effectively without weighting
- Class weighting does not significantly influence performance

---

### Conclusion

Class weights are not necessary for this dataset and can be safely removed without impacting model performance.

---

## Step 11.4 — Ablation 3: Removal of Normalization

### Objective

Evaluate the impact of Z-score normalization on model performance and training stability.

---

### Changes Implemented

- Removed normalization from preprocessing pipeline
- Replaced z-score normalization with identity function (raw intensities)

---

### Results Summary

| Model Variant | Normalization | Test Accuracy | ROC-AUC |
|--------------|--------------|--------------|--------|
| Baseline     | Yes          | 99.47%       | 0.9998 |
| Ablation 3   | No           | 99.39%       | 0.9997 |

---

### Observations

- Slight drop in accuracy (~0.08%)
- Minor decrease in ROC-AUC
- Training instability observed:
  - High validation loss in early epochs
  - Fluctuating loss across epochs

---

### Interpretation

- Model is robust to lack of normalization
- Tumor features are strong enough to be learned from raw intensities
- However, normalization significantly improves training stability and convergence

---

### Conclusion

Normalization is not essential for final performance but is critical for stable and reliable training.

---

## Step 12 - Streamlit Decision-Support App

### Objective

Deliver an interactive interface for MRI inference so users can upload a scan, review tumor likelihood, and inspect Grad-CAM explainability outputs without running training scripts.

---

### Step 12.1 - Inference Utilities for App Runtime

File Added:
- `src/inference.py`

Implemented:
- Device-aware model loading (`cpu`/`cuda`) from `checkpoints/best_model.pth`
- Upload-safe NIfTI preprocessing from raw bytes
- Reuse of training-aligned preprocessing:
  - 4D handling
  - z-score normalization
  - valid slice extraction
  - 2.5D stacking
  - resize to `(3, 224, 224)`
- Slice-level probability prediction
- Patient-level score aggregation (top-k mean)
- Single-slice Grad-CAM generation for UI visualization

Outcome:
Inference path is modularized and reusable by the web app.

---

### Step 12.2 - Streamlit Frontend Implementation

File Added:
- `app/streamlit_app.py`

Implemented:
- Streamlit page configuration and custom UI theme
- Sidebar controls:
  - checkpoint path
  - decision threshold
  - Grad-CAM toggle
  - Grad-CAM quality controls (smoothing, clipping, saliency threshold, focus percentile)
- Upload flow for `.nii` / `.nii.gz`
- Cached model loading for fast repeated inference
- Prediction summary panel:
  - patient score
  - threshold
  - decision label
  - valid slice count
- Slice viewer with index slider
- Explainability view with three synchronized panels:
  - MRI slice
  - Grad-CAM on brain mask
  - overlay image
- Probability trend chart across slices

Outcome:
End-to-end interactive decision-support interface is operational.

---

### Step 12.3 - Integration Validation

Validated:
- App loads trained checkpoint successfully
- Uploaded scans pass preprocessing pipeline without structural mismatch
- Slice-level predictions and patient-level decision are produced correctly
- Grad-CAM rendering is integrated into interactive inspection workflow

Run Command:
- `streamlit run app/streamlit_app.py`

---

### Step 12 Summary

Project now includes a complete inference and explainability application layer on top of the trained model pipeline, enabling demonstration-ready MRI decision support.