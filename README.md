# CardioRisk AI

CardioRisk AI is a machine-learning project that studies cardiovascular risk prediction using multiple datasets, preprocessing pipelines, classical ML models, and neural-network classifiers.

The goal of the project is not to present a medical product. The goal is to demonstrate an end-to-end data science workflow: cleaning raw health datasets, preparing model-ready features, training different model families, comparing their outputs, and deploying the result in an interactive Streamlit dashboard.

## What This Project Demonstrates

- Data cleaning and preprocessing for tabular healthcare datasets
- Handling missing values, categorical variables, scaling, and engineered features
- Building classical machine-learning models for classification tasks
- Building neural-network style models for comparison
- Comparing model families across multiple cardiovascular prediction targets
- Turning notebook experiments into a reproducible local VS Code project
- Packaging a machine-learning workflow into a clean Streamlit interface
- Preparing a project for GitHub portfolio presentation

## Prediction Tasks

The dashboard compares models across three related classification problems:

- 10-year coronary heart disease risk
- Heart disease classification
- Stroke risk classification

## Model Comparison

The training script evaluates both model families with ROC-AUC on held-out validation splits.

| Task | Classical ML AUC | Neural Network AUC |
| --- | ---: | ---: |
| Coronary heart disease | 0.724 | 0.667 |
| Heart disease | 0.928 | 0.899 |
| Stroke | 0.811 | 0.584 |

These results are useful for discussing the tradeoffs between classical ML and neural-network approaches on structured tabular data. In this project, the classical models perform strongly, which is a realistic and valuable finding for small-to-medium tabular datasets.

## Project Structure

```text
cardio_risk_project/
|-- app.py                         # Streamlit dashboard and prediction flow
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
`-- requirements.txt
```

## Run Locally

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the dashboard:

```powershell
streamlit run app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

`localhost` means the app is running only on your own computer. It is perfect for development and screenshots, but hiring teams cannot open that link from their devices.

The repository includes lightweight trained artifacts in `models/artifacts/`, so the dashboard can run immediately after dependency installation.

## Deploy For Hiring Teams

For a Streamlit portfolio project, the best option is usually Streamlit Community Cloud because it deploys directly from GitHub and gives you a public `.streamlit.app` link.

Recommended deployment path:

1. Push this project to a public GitHub repository.
2. Go to Streamlit Community Cloud: https://streamlit.io/cloud
3. Sign in with GitHub.
4. Choose the repository, branch, and `app.py` as the main file.
5. Deploy the app.
6. Add the public app link to your GitHub README, CV, and LinkedIn post.

Good public links look like:

```text
https://your-project-name.streamlit.app
```

Do not use `localhost` on your CV or LinkedIn. Use `localhost` only while developing locally.

Alternative deployment options:

- Hugging Face Spaces: good for ML portfolios and public demos.
- Render: good if you want more backend control.
- GitHub Pages: not suitable for this app because Streamlit runs Python on a server.

## Rebuild The Models

To regenerate the model artifacts from the processed datasets:

```powershell
python scripts/train_models.py
```

The script trains:

- Classical ML pipelines for coronary heart disease and stroke
- A random forest model for heart disease
- MLP neural-network classifiers for model-family comparison

## Disclaimer

This project is for educational and portfolio demonstration purposes only. It is not a medical device and should not be used for clinical diagnosis or treatment decisions.
