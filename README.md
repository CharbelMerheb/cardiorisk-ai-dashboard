# CardioRisk AI

CardioRisk AI is an end-to-end machine-learning dashboard for cardiovascular risk prediction. The project combines data preprocessing, feature engineering, classical machine-learning models, neural-network classifiers, model evaluation, and Streamlit deployment into one reproducible workflow.

The dashboard compares model outputs across three cardiovascular prediction tasks:

- 10-year coronary heart disease risk
- Heart disease classification
- Stroke risk classification

## Live Demo

Access the deployed dashboard here: https://cardiorisk-ai-dashboard.streamlit.app/

## Features

- Interactive Streamlit dashboard for model inference
- Side-by-side comparison between classical ML and neural-network classifiers
- Preprocessed tabular healthcare datasets
- Reproducible model training script
- Model-ready feature inspection inside the app
- Lightweight saved model artifacts for immediate local execution

## Technical Workflow

1. Raw healthcare datasets are cleaned and transformed in preprocessing notebooks.
2. Missing values, categorical variables, scaling, and engineered features are handled for model training.
3. Classical ML and neural-network classifiers are trained on processed datasets.
4. Models are evaluated using ROC-AUC on held-out validation splits.
5. Trained artifacts are loaded by a Streamlit dashboard for interactive prediction.

## Model Performance

| Task | Classical ML AUC | Neural Network AUC |
| --- | ---: | ---: |
| Coronary heart disease | 0.724 | 0.667 |
| Heart disease | 0.928 | 0.899 |
| Stroke | 0.811 | 0.584 |

The comparison highlights an important pattern in structured tabular data: classical machine-learning models can perform competitively, and sometimes better, than neural-network approaches on smaller datasets.

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- Streamlit
- joblib
- Jupyter notebooks

## Project Structure

```text
cardio_risk_project/
|-- app.py                         # Streamlit dashboard and prediction flow
|-- requirements.txt               # Python dependencies
|-- scripts/
|   `-- train_models.py             # Rebuilds model artifacts from processed data
|-- data/
|   |-- Original_Datasets/          # Raw reference datasets
|   `-- Processed_Datasets/         # Cleaned model-ready datasets
|-- models/
|   |-- data_preprocessing/         # Data cleaning and preprocessing notebooks
|   |-- ml/                         # Classical ML notebooks
|   |-- dl/                         # Deep-learning notebooks
|   `-- artifacts/                  # Lightweight trained .joblib models
```

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

Run the Streamlit dashboard:

```powershell
streamlit run app.py
```

The app will open in the browser at a local Streamlit URL, usually:

```text
http://localhost:8501
```

## Rebuilding Models

The repository includes lightweight trained artifacts in `models/artifacts/`. To regenerate them from the processed datasets:

```powershell
python scripts/train_models.py
```

The training script creates:

- Classical ML pipelines for coronary heart disease and stroke
- A random forest classifier for heart disease
- MLP neural-network classifiers for model-family comparison

## Disclaimer

This project is for educational and portfolio demonstration purposes only. It is not a medical device and should not be used for clinical diagnosis or treatment decisions.
