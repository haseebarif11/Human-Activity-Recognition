# Human Activity Recognition (HAR) - Hybrid ML/DL Pipeline

A clean, end-to-end Human Activity Recognition project built in Python and designed for Kaggle workflows.  
The notebook combines deep learning feature extraction (EfficientNetB0) with classical machine learning and ensembling to classify human actions from images.

## Project Highlights

- Hybrid architecture using EfficientNetB0 + traditional ML classifiers.
- Multi-model benchmarking (including Random Forest and XGBoost).
- End-to-end training, evaluation, visualization, and artifact export.
- Kaggle-ready data paths and GPU-aware setup.
- Includes extracted result figures from the notebook for quick repo preview.

## Repository Structure

```text
.
├── assets/
│   └── images/
│       ├── training-plot-01.png
│       ├── training-plot-02.png
│       └── training-plot-03.png
├── notebooks/
│   └── har-lab-proj-final.ipynb
├── .gitignore
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── requirements.txt
```

## Visual Results

### Training/Evaluation Figure 1
![Training Plot 1](assets/images/training-plot-01.png)

### Training/Evaluation Figure 2
![Training Plot 2](assets/images/training-plot-02.png)

## Tech Stack

- Python
- TensorFlow / Keras (`EfficientNetB0`)
- Scikit-learn
- XGBoost
- OpenCV
- NumPy, Pandas, Matplotlib, Seaborn
- Joblib

## Quick Start

### 1) Clone

```bash
git clone <your-repo-url>
cd Machine-learning--main
```

### 2) Create Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install Dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the Notebook

Open and run:

`notebooks/har-lab-proj-final.ipynb`

## Workflow Summary

1. Authenticate/download datasets via `kagglehub`.
2. Load and preprocess labeled training data.
3. Extract deep features with EfficientNetB0.
4. Train/evaluate multiple ML models on extracted features.
5. Build a hybrid/meta model and compare performance.
6. Save trained artifacts (model weights, scaler, label encoder, plots).

## Notes

- This notebook is optimized for Kaggle paths (for example `/kaggle/input` and `/kaggle/working`).
- If running locally, update data paths accordingly.
- A future improvement is to save Keras models in `.keras` format instead of legacy `.h5`.

## Contributing

Contributions are welcome. See `CONTRIBUTING.md` for development and pull request guidelines.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
