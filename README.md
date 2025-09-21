# Aquaculture Health Predictor

A Streamlit web app and accompanying analysis scripts for exploring aquaculture environmental data and predicting microbiome health scores for various farm types and locations (e.g., Lakshadweep). The app demonstrates feature engineering and trains a regression model to estimate a farm's microbiome health score based on water quality and contextual variables.

Note: The Streamlit app (`app.py`) currently generates a synthetic dataset for demo purposes so it can run out-of-the-box. The notebook-style analysis script (`Aquaculture Analysis.py`) expects a local CSV that is not part of this repository; see Dataset section for details.

## Features
- Interactive Streamlit UI with tabs:
  - Single prediction with sliders and dropdowns
  - Batch prediction via CSV upload and downloadable results
  - Model analysis (R², MSE, plots, feature importance when available)
  - Data exploration (summary stats and visualizations)
- Feature engineering for domain signals (e.g., stress index, nutrient balance, seasonal components, water quality index)
- Scikit-learn pipeline with preprocessing (scaling, one-hot encoding) and Gradient Boosting/Random Forest/XGBoost experiments
- Ready-to-download predictions for batch inputs

## Directory Structure
```
Aquaculture Analysis/
├─ app.py                     # Streamlit application (synthetic demo data)
├─ Aquaculture Analysis.py    # Exploratory/ML analysis (expects external CSV)
├─ hi.py                      # Scratch Python snippets (not part of app)
├─ linkedlist.c               # Scratch C snippet (not part of app)
├─ g.csv                      # Empty placeholder file
├─ README.md                  # This documentation
├─ requirements.txt           # Python dependencies
└─ .gitignore                 # Git ignore rules
```

## Quickstart
### 1) Create and activate a virtual environment (recommended)
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the Streamlit app
```bash
streamlit run app.py
```
The app will open in your browser (usually http://localhost:8501).

## Usage
### Single Prediction
- Open the app and use the controls in the "Single Prediction" tab.
- Click "Predict Health Score" to see the predicted value and a qualitative status.

### Batch Prediction
- Prepare a CSV with the following columns (header row required):
  - `location`, `farm_type`, `temperature`, `salinity`, `dissolved_oxygen`, `ph`, `nitrates`, `phosphates`, `ammonia`, `chlorophyll`, `turbidity`
- Upload it in the "Batch Prediction" tab.
- Download the returned CSV containing `predicted_health_score` and `health_status`.

## Dataset
- The Streamlit app uses synthetic data generated in-memory for demonstration, so it does not require any external files.
- The exploratory analysis script `Aquaculture Analysis.py` references a CSV at a local path similar to:
  `C:/Users/.../Aquaculture_Project/lakshadweep_aquaculture_data_2023_enhanced.csv`
  This file is not included. To run that script, edit the path(s) in the script to point to your dataset location, or place the CSV alongside the script and use a relative path, e.g.:
  ```python
  df = pd.read_csv('lakshadweep_aquaculture_data_2023_enhanced.csv')
  ```

## Modeling Overview
- Target variable: `microbiome_health_score`
- Example features include:
  - Categorical: `location`, `farm_type`
  - Numeric: `temperature`, `salinity`, `dissolved_oxygen`, `ph`, `nitrates`, `phosphates`, `ammonia`, `chlorophyll`, `turbidity`
  - Engineered: `temp_oxygen_interaction`, `nutrient_balance`, `stress_index`, `month_sin`, `month_cos`, `total_nutrients`, `water_quality_index`
- Preprocessing: `StandardScaler` for numeric and `OneHotEncoder` for categorical features via `ColumnTransformer`
- Example regressors explored: `GradientBoostingRegressor`, `RandomForestRegressor`, `XGBRegressor`, `MLPRegressor`, etc.

## Requirements
See `requirements.txt` for the exact list. Main libraries:
- streamlit, pandas, numpy
- scikit-learn, xgboost
- seaborn, matplotlib
- joblib
- shap (used in analysis script for explainability; optional for the Streamlit app)

## Troubleshooting
- If the app fails to import `xgboost` or `shap`, and you only intend to run the app (not the analysis script), you can remove those packages from `requirements.txt` or install them:
  ```bash
  pip install xgboost shap
  ```
- If plots do not display in notebooks, ensure you call `plt.show()` after each plot.
- On Windows PowerShell, if virtual env activation is blocked, run:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

## Contributing
- Open an issue or submit a pull request in the GitHub repository.
- For larger changes, please discuss via an issue first.
