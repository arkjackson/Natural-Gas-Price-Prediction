# Natural-Gas-Price-Prediction

This repository contains a machine learning pipeline to forecast natural gas prices for the next 1–3 months using XGBoost with engineered time series features and automated hyperparameter tuning.

## 📁 Project Structure
```
NATURAL-GAS-PRICE-PREDICTION/
├── config/ # Environment settings (e.g., constants, paths)
├── data/ # Raw/processed data and preprocessing module
├── models/ # Model training and tuning logic
├── utils/ # Utility functions (e.g., evaluation metric)
├── main.py/ # Model training script
├── inference.py/ # Inference script on test set 
├── requirements.txt/ # Required Python packages
```

## 📊 Sample Data Format

| date       | price | price\_t+1 | price\_t+2 | price\_t+3 |
| ---------- | ----- | ---------- | ---------- | ---------- |
| 2003-09-30 | 4.62095  | 4.63391   | 4.49167    | 6.14     |

- `price_t+1`, `price_t+2`, `price_t+3`: Future prices (1, 2, 3 months ahead)

## 🧠 Modeling Overview

- **Algorithm**: XGBoost + MultiOutputRegressor
- **Target Variables**: 'price_t+1', 'price_t+2', 'price_t+3'
- **Feature Engineering**:
    - Lag features
    - Rolling statistics (mean & std)
    - Month dummies (seasonality)
    - Log-transformed features
    - Feature selection with 'SelectKBest'
- **Hyperparameter Optimization**: [Optuna](https://optuna.org/)
- **Evaluation Metric**: MAPE (Mean Absolute Percentage Error)

## ⚙️ Training Pipeline

1. Load raw CSV data and sort by date
2. Generate time-series features and perform seasonal decomposition
3. Select top-K features via correlation with `price_t+1`
4. Split data into train/validation sets (time-based split)
5. Optimize XGBoost hyperparameters via Optuna
6. Train MultiOutputRegressor with best parameters
7. Save model, scalers, and feature column list

## Result

- Metrics

| model   | MAE   | RMSE | MAPE |
| --------| ----- | ---- | -----|
| XGBoost | 0.93  | 1.39 | 20.94|

- Pattern Analysis

![natural gas price](./results/img/natural%20gas%20price.png)

![mae prediction](./results/img/mae%20prediction.png)


-  MAE values surge as natural gas prices soar in first half, mid-year 2022
    - 2022-02: Russia-Ukraine War (European Gas Supply Disruptions)
    - 2022-06: Freeport LNG Explosion (Closing 15 million tonnes of facility annually)

## 📌 TODO

- [ ] Integrate event-driven features using text/news data