# Natural-Gas-Price-Prediction

This project predicts future natural gas prices using machine learning models (CatBoost, LightGBM, XGBoost).

## 📁 Project Structure

- `config/`: Environment setting
- `data/`: Train, Test data (CSV format) and Preprocessing module
- `models/`: Training model modules
- `utils/`: Utility modules (metrics)
- `train.py`: Training model script
- `inference.py`: Inference script (test set prediction)
- `requirements.txt`: Required package

## 📊 Sample Data Format

| date       | price | price\_t+1 | price\_t+2 | price\_t+3 |
| ---------- | ----- | ---------- | ---------- | ---------- |
| 2003-09-30 | 2.55  | 2.63       | 2.70       | 2.68       |

## 🧠 Model Overview

- Algorithm: XGBoost + MultiOutputRegressor
- Target: price_t+1, price_t+2, price_t+3 (1–3 month ahead forecasts)
- Features:
    - Lag variables
    - Rolling means & standard deviations
    - Seasonal decomposition
    - Log-transformed variables
- Evaluation Metric: MAPE (Mean Absolute Percentage Error)

## 📌 TODO

- Add text/news based event feature