# Brain MRI AI Decision Support — Project Log

## Step 1 — Project Setup

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

# Step 3 — Build Training Dataset

**Goal:** Prepare MRI data for model training by organizing the **OASIS (normal)** and **BraTS (abnormal)** datasets and creating training, validation, and test splits compatible with the PyTorch training pipeline.

---

## Step 3.1 — Dataset Organization

**Goal:** Collect MRI volumes from the OASIS and BraTS datasets and organize them for model training.

### Actions

• Defined dataset paths for **OASIS Clean Data** and **BraTS 2020 Training Data** directories  
• Collected MRI file paths from the OASIS dataset representing **normal brain scans**  
• Traversed BraTS patient directories and selected the **T1CE MRI modality** for tumor detection  
• Created separate lists of MRI file paths for **normal (OASIS)** and **abnormal (BraTS)** datasets

**Outcome:** MRI volumes from both datasets were successfully identified and organized into lists representing normal and abnormal brain scans.

---

## Step 3.2 — Dataset Splitting

**Goal:** Divide the collected MRI volumes into training, validation, and test subsets.

### Actions

• Randomly shuffled the MRI file lists to prevent ordering bias  
• Implemented dataset splitting using the following ratio:
70% Training
15% Validation
15% Test
• Applied the split separately to the **OASIS (normal)** and **BraTS (abnormal)** datasets  
• Generated file lists corresponding to each dataset partition

**Outcome:** MRI volumes were successfully divided into **training, validation, and test subsets**, ensuring proper data separation for model training and evaluation.

---

## Step 3.3 — PyTorch Dataset Integration

**Goal:** Enable dynamic loading and preprocessing of MRI volumes during training.

### Actions

• Implemented a custom **PyTorch Dataset class (`MRIDataset`)**  
• Loaded MRI volumes directly from `.nii` files using **NiBabel**  
• Generated **2.5D slices** by stacking adjacent slices _(i-1, i, i+1)_  
• Applied preprocessing including **resizing to 224 × 224 pixels** and **normalization**  
• Converted slices into tensors compatible with **ResNet50 input requirements**

**Outcome:** MRI volumes can now be dynamically loaded and converted into **CNN-ready 2.5D slices during training**, eliminating the need to store intermediate image files and enabling efficient data processing.

---

# Step 4 — Model Development

**Goal:** Build and train a deep learning model capable of classifying MRI scans as **normal or tumor**.

---

## Step 4.1 — Define CNN Architecture

**Goal:** Implement a deep learning model capable of learning features from MRI slices using transfer learning.

### Actions

• Implemented **ResNet50 architecture using PyTorch**  
• Loaded **ImageNet pretrained weights** for transfer learning  
• Modified the **final fully connected layer** to output **2 classes (Normal / Tumor)**  
• Ensured compatibility with **224 × 224 × 3 MRI slice inputs**

**Outcome:** A customized **ResNet50 model** capable of performing MRI classification.

---

## Step 4.2 — Define Training Components

**Goal:** Configure the core components required for model training.

### Actions

• Implemented **CrossEntropyLoss** for classification  
• Configured **Adam optimizer** for gradient descent  
• Defined **learning rate and training hyperparameters**  
• Enabled **automatic device selection (CPU / GPU)**

**Outcome:** Training configuration prepared with **loss function, optimizer, and device setup**.

---

## Step 4.3 — Implement Training Loop

**Goal:** Train the CNN model using MRI data batches.

### Actions

• Loaded MRI datasets using **PyTorch DataLoader**  
• Implemented training loop with:

Forward pass

Loss calculation

Backpropagation

Optimizer updates

• Tracked **training loss and accuracy per epoch**

**Outcome:** The CNN model successfully trains on MRI slices generated from **OASIS (normal)** and **BRATS (tumor)** datasets.

---

## Step 4.4 — Data Preparation for Training

**Goal:** Integrate MRI datasets and preprocessing pipeline into the training workflow.

### Actions

• Loaded **OASIS dataset** as the **normal brain class**  
• Loaded **BRATS dataset (FLAIR modality)** as the **tumor class**  
• Implemented **2.5D slicing strategy** by stacking adjacent slices _(i−1, i, i+1)_  
• Resized slices to **224 × 224 pixels**  
• Normalized slice intensity values

**Outcome:** MRI volumes are converted into **CNN-ready 2.5D slices compatible with ResNet50 input requirements**.

---

## Step 4.5 — Model Training Execution

**Goal:** Train the CNN model using the prepared dataset.

### Actions

• Executed **model training for 5 epochs**  
• Monitored **training loss and accuracy during training**  
• Observed **rapid convergence due to transfer learning**

### Training Results

Epoch 1/5
Train Loss: 12.0793
Train Accuracy: 0.9711

Epoch 2/5
Train Loss: 0.8738
Train Accuracy: 1.0000

Epoch 3/5
Train Loss: 0.4563
Train Accuracy: 1.0000

Epoch 4/5
Train Loss: 0.2839
Train Accuracy: 1.0000

Epoch 5/5
Train Loss: 0.1378
Train Accuracy: 1.0000

**Outcome:** The model successfully learned MRI classification patterns and achieved **high training accuracy**.

---

## Step 4.6 — Save Trained Model

**Goal:** Persist the trained CNN model for later inference and evaluation.

### Actions

• Saved trained model weights using **PyTorch `torch.save()`**  
• Stored model checkpoint inside the **models/** directory

**Outcome:** The model successfully learned MRI classification patterns and achieved **high training accuracy**.

---

## Step 4.6 — Save Trained Model

**Goal:** Persist the trained CNN model for later inference and evaluation.

### Actions

• Saved trained model weights using **PyTorch `torch.save()`**  
• Stored model checkpoint inside the **models/** directory

models/mri_resnet50.pth

**Outcome:** The trained CNN model is saved and available for **inference, evaluation, and deployment**.

---

## Step 5 — Explainable AI

Goal:  
Provide visual explanations of model predictions to support medical interpretability and help understand which regions of the MRI influence the model's decision.

---

### Step 5.1 — Implement Grad-CAM

Goal:  
Highlight regions of the MRI image that influenced the model's prediction.

Actions:

- Implemented the Grad-CAM (Gradient-weighted Class Activation Mapping) technique for CNN interpretability.
- Attached forward and backward hooks to the final convolutional layer (`layer4`) of the ResNet50 model.
- Captured feature maps produced during the forward pass.
- Computed gradients of the predicted class with respect to these feature maps.
- Calculated channel-wise importance weights using global average pooling of gradients.
- Generated a class activation map by combining the feature maps with the computed weights.
- Applied ReLU activation and normalization to produce the final Grad-CAM heatmap.

Outcome:
Grad-CAM successfully generates heatmaps that highlight the most influential regions used by the model during prediction, improving transparency and interpretability of the CNN model.

---

### Step 5.2 — Overlay Heatmap on MRI Image

Goal:  
Visualize Grad-CAM results by overlaying activation heatmaps directly on MRI images.

Actions:

- Resized the Grad-CAM heatmap to match the MRI input resolution (224 × 224).
- Converted the normalized heatmap into a colored activation map using the JET color scheme.
- Blended the activation heatmap with the original MRI image using weighted transparency.
- Displayed the resulting visualization using Matplotlib for easier inspection.

Outcome:
The final visualization clearly shows highlighted regions of the MRI image where the model focused while making predictions, providing an intuitive explanation of the model’s decision process.

---
