# Synapse X

![Build](https://img.shields.io/badge/build-not%20configured-lightgrey)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45.1-FF4B4B?logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

Synapse X is a brain MRI decision-support project for binary classification of neurological scans into healthy and abnormal cases. It combines MRI preprocessing, 2.5D slice stacking, multi-modal model support, and patient-level score aggregation to produce interpretable predictions from volumetric NIfTI scans. The project is designed for research workflows, model experimentation, and interactive review through a Streamlit interface.

## Key Features

- MRI volume loading, normalization, resampling, slice filtering, and 2.5D stacking.
- Support for single-volume and multi-modal inference workflows.
- Patient-level prediction aggregation using top-k slice scoring.
- Class-weighted training with validation tracking and best-checkpoint saving.
- Streamlit dashboard for upload, threshold tuning, slice browsing, probability trends, and Grad-CAM visualization.
- Threshold calibration and dataset utility scripts for evaluation and maintenance.

## Tech Stack

- Language: Python 3.11.
- Deep Learning: PyTorch, TorchVision, TorchAudio.
- Scientific Computing: NumPy, SciPy, scikit-learn, Pandas.
- Medical Imaging: NiBabel, OpenCV.
- Visualization and UI: Streamlit, Matplotlib, PIL.
- Reporting and Export: ReportLab.
- Tooling: `requirements.txt`, `runtime.txt`, PowerShell-friendly Windows workflow.

## Project Structure

```text
.
├── app/
│   └── streamlit_app.py          # Interactive Streamlit UI
├── checkpoints/
│   └── best_model.pth            # Saved model checkpoint
├── data/
│   ├── raw/                      # BRATS, OASIS, IXI, and other source data
│   ├── preprocessed/             # Cached preprocessed volumes
│   └── splits/                   # Patient-level split JSON files
├── models/
│   └── model.py                  # CNN and multi-modal architectures
├── outputs/                      # Calibration results and reports
├── results/                      # Generated visualizations such as Grad-CAM outputs
├── scripts/                      # Dataset, calibration, and audit utilities
├── src/
│   ├── data/                     # Dataset and DataLoader helpers
│   ├── preprocessing/            # NIfTI loading, normalization, resize, slice extraction
│   ├── utils/                    # Grad-CAM and reproducibility helpers
│   └── inference.py              # Preprocessing and prediction helpers
├── train.py                      # Training and evaluation entry point
├── visualize_gradcam.py          # Grad-CAM visualization utility
└── requirements.txt              # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.11, matching the version pinned in `runtime.txt`.
- `pip` and a virtual environment tool such as `venv`.
- Sufficient disk space for MRI datasets and generated checkpoints.
- Optional CUDA-capable GPU for faster training and inference.
- MRI datasets placed in the expected local paths under `data/raw/`.
- No required environment variables are currently defined in the codebase.

### Installation

1. Clone the repository.

   ```bash
   git clone https://github.com/Shayan-Bhowmik/image-processing-mri
   cd image-processing-mri
   ```

2. Create and activate a virtual environment.

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install the project dependencies.

   ```bash
   pip install -r requirements.txt
   ```

4. Place the datasets in the expected locations if you plan to train or evaluate locally.

   - BRATS 2020 training data: `data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData`
   - BRATS 2021 data: `data/raw/brats2021_extracted` or the equivalent local extraction path used by your split files
   - OASIS healthy scans: `data/raw/oasis/OASIS_Clean_Data/OASIS_Clean_Data`

### Environment Setup

There is no required `.env` file in the current implementation. If you add local-only settings later, keep them in a private `.env` file and do not commit secrets or machine-specific paths. The repository already includes a Streamlit config file at `.streamlit/config.toml` to make the app behave well on Windows.

### Run Locally

```bash
python train.py
```

```bash
streamlit run app/streamlit_app.py
```

## Usage

Train the model with the default patient split:

```bash
python train.py --epochs 20
```

Run evaluation only from the saved checkpoint:

```bash
python train.py --eval-only
```

Calibrate a patient-level decision threshold:

```bash
python scripts/calibrate_threshold.py --split-name val
```

Launch the interactive dashboard and upload a `.nii` or `.nii.gz` scan:

```bash
streamlit run app/streamlit_app.py
```

Example Python inference workflow:

```python
from src.inference import (
    aggregate_patient_score,
    load_trained_model,
    predict_slices,
    preprocess_uploaded_nifti,
)

model, device = load_trained_model("checkpoints/best_model.pth")

with open("case.nii.gz", "rb") as f:
    payload = f.read()

prep = preprocess_uploaded_nifti(payload, "case.nii.gz")
slice_predictions, slice_probabilities = predict_slices(model, prep["input_batch"], device)
patient_score = aggregate_patient_score(slice_probabilities, top_k=10)
```

## API Reference

This repository does not expose a public HTTP API. The core inference surface is implemented as Python functions in `src/inference.py`.

- `load_trained_model(checkpoint_path, in_channels, num_classes)`: loads a saved checkpoint and returns a ready-to-use model plus device.
- `preprocess_uploaded_nifti(uploaded_bytes, uploaded_filename, ...)`: converts a single uploaded NIfTI file into model-ready tensors.
- `preprocess_uploaded_multimodal_nifti(uploaded_by_modality, ...)`: builds a multi-modal input batch from multiple uploaded scans.
- `predict_slices(model, input_batch, device)`: returns slice-level class predictions and positive-class probabilities.
- `aggregate_patient_score(slice_probs, top_k=10)`: converts slice probabilities into one patient-level score.
- `build_gradcam_for_slice(...)`: generates a Grad-CAM heatmap for an individual preprocessed slice.

Key CLI entry points:

- `train.py`: training and evaluation, with flags such as `--epochs`, `--eval-only`, `--split-path`, and `--exclude-brats2021`.
- `scripts/calibrate_threshold.py`: threshold calibration for held-out validation or test splits.

## Screenshots / Demo

Add screenshots or a short GIF here showing the Streamlit dashboard, slice viewer, probability trend, Grad-CAM overlay, and download outputs.

## Contributing

1. Fork the repository and create a feature branch.
2. Make focused changes with clear commits.
3. Run the relevant training, inference, or Streamlit checks locally.
4. Open a pull request with a short summary of the change and any validation results.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.

## Live Demo

**Hosted at:** https://synapse-x-brain-mri-imaging.streamlit.app/

Visit the live version of the project at the link above.