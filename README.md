# Time Series Forecasting & Predictive Threshold Estimation using LSTM

A deep learning project implementing a Long Short-Term Memory (LSTM) network to model, predict, and analyze temporal sensor readings, estimate threshold breach horizons (T-prediction), and evaluate forecasting accuracy across individual and aggregated sensor sequences.

---

## Project Overview

Time-series observations collected from distributed sensors often exhibit non-linear dependencies and long-term trends. This project implements:

- **LSTM-Based Sequential Modeling**: Mitigates vanishing gradient issues of standard RNNs to retain long-range temporal dependencies.
- **Sliding-Window Transformation**: Restructures raw sequential records into a supervised learning problem (X -> Y).
- **T-Prediction Algorithm**: Computes the minimum future time step T at which a forecasted value increases by a specified threshold percentage p:

  Target Value = (1 + p) * x(t) = 1.10 * x(t)
  Condition: x(t + T) >= (1 + p) * x(t)

---

## Model Architecture

The network pipeline is developed in TensorFlow/Keras and features:

- **Input Dimension**: Sequential window of historical sensor observations.
- **LSTM Layer 1**: 50 hidden units with `return_sequences=True`.
- **Dropout Layer 1**: Rate = 0.2 to prevent overfitting.
- **LSTM Layer 2**: 25 hidden units with `return_sequences=False`.
- **Dropout Layer 2**: Rate = 0.2.
- **Dense Layers**: Fully connected layers refining the extracted temporal features into forecast values.
- **Optimizer & Loss**: Adam optimizer minimizing Mean Squared Error (MSE) loss.
- **Regularization**: Early Stopping monitoring validation loss with a patience threshold.

---

## Data Preprocessing & Pipeline

1. **Cleaning**: Handles missing and null sensor entries.
2. **Timestamp Alignment**: Normalizes timestamps relative to the initial recorded observation per sensor.
3. **Feature Scaling**: Maps values to the [0, 1] range using `MinMaxScaler`.
4. **Supervised Framing**: Constructs sliding window sequences with configurable input lengths and future forecasting steps.
5. **Evaluation Metrics**: Models are assessed using:
   - **MAE** (Mean Absolute Error)
   - **MSE** (Mean Squared Error)
   - **RMSE** (Root Mean Squared Error)
   - **R² Score** (Coefficient of Determination)

---

## Experimental Results

### 1. Global / Aggregated Sequence Evaluation (80% Train / 20% Test)

| Metric | Evaluation Result |
| :--- | :--- |
| **R² Score** | **0.9964** |
| **Mean Absolute Error (MAE)** | **0.5614** |
| **Root Mean Squared Error (RMSE)** | **1.8717** |
| **Mean Squared Error (MSE)** | **3.5032** |
| **Time to 10% Increase (T)** | **54 steps** |

### 2. Individual Sensor Evaluation Highlights

- **Sensor 8 (High Precision)**: Achieved R² = 0.9857, MAE = 0.9681, and MSE = 2.6642, with prediction error centered closely around zero.
- **Threshold Crossing (p = 0.10 / 10% Increase)**: Accurately determined onset times ranging from T = 21 steps (Sensor 3) to T = 309 steps (Sensor 4), correctly outputting -1 for signals that never breached the target increase within the forecast horizon.

---

## Repository Structure

```text
├── data.csv                # Raw multi-sensor sequential dataset
├── Predict_T.ipynb         # Model training, evaluation, and T-prediction notebook
├── LSTM_Prediction.pdf     # Project report documentation
├── .gitignore              # Ignored files and directories
└── README.md               # Project documentation

