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