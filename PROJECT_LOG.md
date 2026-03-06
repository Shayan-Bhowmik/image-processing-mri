# Brain MRI AI Decision Support — Project Log

## Step 1 — Project Setup (Completed)

**Goal:** Create the working environment and initialize the project structure for the Brain MRI AI project.

---

## Step 1.1 — Create Project Repository

**Goal:** Initialize the project directory and organize the development structure.

### Actions

- Created the main project directory **brain-mri-ai**
- Organized a modular folder structure to support the ML pipeline
- Added directories for data storage, preprocessing, model development, explainability, and application interface
- Created folders for experimentation and storing trained models
- Added **requirements.txt** to manage project dependencies

**Outcome:** Project repository initialized with a structured and scalable directory layout ready for development.

---

## Step 1.2 — Install Required Libraries

**Goal:** Set up the Python environment with all necessary dependencies.

### Actions

- Defined required libraries for deep learning, data processing, visualization, MRI handling, and web deployment
- Added all dependencies to **requirements.txt**
- Installed the required libraries using the requirements file

**Outcome:** Development environment configured successfully with all required packages installed.

---

## Step 1.3 — Git Hygiene

**Goal:** Maintain a clean and efficient repository by managing tracked files properly.

### Actions

- Implemented Git hygiene practices to maintain repository cleanliness
- Created a **.gitignore** file to prevent unnecessary files from being tracked
- Excluded large MRI datasets and generated files from version control
- Ignored system files, cache files, and temporary artifacts
- Ensured only essential source code, configuration files, and documentation are committed

**Outcome:** Repository remains lightweight, organized, and suitable for version control and collaboration.

---

# Step 2 — MRI Data Handling

**Goal:** Prepare MRI data for model training by loading MRI volumes, extracting informative slices, and converting them into CNN-ready images.

---

## Step 2.1 — Load MRI Using NiBabel

**Goal:** Load `.nii` MRI files and convert them into NumPy arrays for further processing.

### Actions

• Implemented MRI loading using the **NiBabel** library
• Loaded `.nii` MRI files from the dataset directory
• Converted MRI volumes into NumPy arrays
• Verified successful loading by printing the MRI volume shape
• Displayed a sample brain slice for visual inspection

**Outcome:** MRI volumes successfully loaded and verified with shape **(240, 240, 155)**, and a brain slice image displayed confirming correct MRI loading.

---

## Step 2.2 — Extract MRI Slices Using 2.5D Slicing

**Goal:** Convert 3D MRI volumes into informative 2D slices while preserving spatial context.

### Actions

• Processed MRI volumes with shape **(240, 240, 155)**
• Extracted slices using the indexing method `volume[:, :, slice_index]`
• Selected **central slices (20–40)** where brain structures are most visible
• Avoided edge slices due to limited anatomical information
• Implemented **2.5D slicing** by stacking adjacent slices _(i-1, i, i+1)_
• Combined the slices to create a **3-channel contextual image**

**Outcome:** Each MRI volume is converted into multiple **context-aware 2.5D slices**, preserving spatial information between neighboring slices while remaining compatible with CNN models.

---

## Step 2.3 — Preprocess MRI Slice

**Goal:** Transform extracted MRI slices into a format suitable for CNN models.

### Actions

• Resized MRI slice images to **224 × 224 pixels**
• Normalized pixel intensity values
• Ensured images follow **3-channel input format**
• Prepared slices to match **ResNet50 input requirements**

**Outcome:** MRI slices are converted into **CNN-ready images (224 × 224 × 3)** suitable for deep learning model training and inference.

---
