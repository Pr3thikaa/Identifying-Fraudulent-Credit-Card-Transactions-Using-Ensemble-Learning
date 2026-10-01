![CI](https://github.com/PrethikaaPRV/Identifying-Fraudulent-Credit-Card-Transactions-Using-Ensemble-Learning/actions/workflows/ci.yml/badge.svg)

# Identifying Fraudulent Credit Card Transactions Using Ensemble Learning

Detects fraudulent credit card transactions using a CatBoost + CNN ensemble,
with class-imbalance handling (SMOTE) and a Streamlit app for real-time prediction.

## Problem
Fraud is extremely rare (~0.17% of transactions in this dataset), so accuracy
alone is misleading. Models are evaluated on fraud-class precision, recall and F1.

## Results (fraud class, test set)
| Metric    | Value |
|-----------|-------|
| Precision | [add] |
| Recall    | [add] |
| F1-score  | [add] |
| Accuracy  | 97%+  |

## Project Structure
- **M1** - Data preprocessing and EDA
- **M2** - Model training (CatBoost + CNN)
- **M3** - Streamlit app
- **src/** - Reusable preprocessing code
- **tests/** - Unit tests (pytest)
- **.github/workflows/** - CI pipeline

## Dataset
Not included due to size. Download from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place `creditcard.csv` inside the `M1/` folder.

## How to Run
1. Download the dataset and place it in `M1/`
2. Run `M1_Data_Preprocessing.ipynb`
3. Run `M2/Catboost_Model_CCF_V2.ipynb`
4. Start the app: `streamlit run M3/App.py`

## Testing & CI
Unit tests (pytest) cover the preprocessing module, and a GitHub Actions
workflow runs them on every push.

    pip install -r requirements.txt
    python -m pytest


\## How to Run

1\. Download the dataset and place in 'M1/'

2\. Run 'M1\_Data\_Preprocessing.ipynb'

3\. Run 'M2/Catboost\_Model\_CCF\_V2.ipynb'

4\. Run the app: 'streamlit run M3/App.py'

