\# Time Series Forecasting \& Predictive Threshold Estimation using LSTM



A deep learning project implementing a Long Short-Term Memory (LSTM) network to model, predict, and analyze temporal sensor readings, estimate threshold breach horizons ($T$-prediction), and evaluate forecasting accuracy across individual and aggregated sensor sequences\[cite: 25, 28, 34].



\---



\## 📌 Project Overview



Time-series observations collected from distributed sensors often exhibit non-linear dependencies and long-term trends\[cite: 28]. This project implements:

\- \*\*LSTM-Based Sequential Modeling\*\*: Mitigates vanishing gradient issues of standard RNNs to retain long-range temporal dependencies\[cite: 25, 28].

\- \*\*Sliding-Window Transformation\*\*: Restructures raw sequential records into a supervised learning problem ($X \\rightarrow Y$)\[cite: 28].

\- \*\*$T$-Prediction Algorithm\*\*: Computes the minimum future time step $T$ at which a forecasted value increases by a specified threshold percentage $p$:

&#x20; $$x(t + T) \\ge (1 + p) \\cdot x(t)$$

\[cite: 34]



\---



\## 🏗️ Model Architecture



The network pipeline is developed in TensorFlow/Keras and features\[cite: 31, 40]:

\- \*\*Input Dimension\*\*: Sequential window of historical sensor observations\[cite: 28, 31].

\- \*\*LSTM Layer 1\*\*: 50 hidden units with `return\_sequences=True`\[cite: 31].

\- \*\*Dropout Layer 1\*\*: Rate = 0.2 to prevent overfitting\[cite: 31, 40].

\- \*\*LSTM Layer 2\*\*: 25 hidden units with `return\_sequences=False`\[cite: 31].

\- \*\*Dropout Layer 2\*\*: Rate = 0.2\[cite: 31, 40].

\- \*\*Dense Layers\*\*: Fully connected layers refining the extracted temporal features into forecast values\[cite: 31].

\- \*\*Optimizer \& Loss\*\*: Adam optimizer minimizing Mean Squared Error (MSE) loss\[cite: 31].

\- \*\*Regularization\*\*: Early Stopping monitoring validation loss with a patience threshold\[cite: 31].



\---



\## ⚙️ Data Preprocessing \& Pipeline



1\. \*\*Cleaning\*\*: Handles missing and null sensor entries\[cite: 28].

2\. \*\*Timestamp Alignment\*\*: Normalizes timestamps relative to the initial recorded observation per sensor\[cite: 28].

3\. \*\*Feature Scaling\*\*: Maps values to the $\[0, 1]$ range using `MinMaxScaler`\[cite: 28, 40].

4\. \*\*Supervised Framing\*\*: Constructs sliding window sequences with configurable input lengths and future forecasting steps\[cite: 28].

5\. \*\*Evaluation Metrics\*\*: Models are assessed using:

&#x20;  - \*\*MAE\*\* (Mean Absolute Error)\[cite: 32]

&#x20;  - \*\*MSE\*\* (Mean Squared Error)\[cite: 33]

&#x20;  - \*\*RMSE\*\* (Root Mean Squared Error)\[cite: 33]

&#x20;  - \*\*$R^2$ Score\*\* (Coefficient of Determination)\[cite: 33]



\---



\## 📊 Experimental Results



\### 1. Global / Aggregated Sequence Evaluation (80% Train / 20% Test)\[cite: 39]



| Metric | Evaluation Result |

| :--- | :--- |

| \*\*$R^2$ Score\*\* | \*\*0.9964\*\*\[cite: 39] |

| \*\*Mean Absolute Error (MAE)\*\* | \*\*0.5614\*\*\[cite: 39] |

| \*\*Root Mean Squared Error (RMSE)\*\* | \*\*1.8717\*\*\[cite: 39] |

| \*\*Mean Squared Error (MSE)\*\* | \*\*3.5032\*\*\[cite: 39] |

| \*\*Time to 10% Increase ($T$)\*\* | \*\*54 steps\*\*\[cite: 39] |



\### 2. Individual Sensor Evaluation Highlights\[cite: 35, 36, 37, 38]

\- \*\*Sensor 8 (High Precision)\*\*: Achieved $R^2 = 0.9857$, $\\text{MAE} = 0.9681$, and $\\text{MSE} = 2.6642$, with prediction error centered closely around zero\[cite: 36, 37].

\- \*\*Threshold Crossing ($p = 0.10$)\*\*: Accurately determined onset times ranging from $T = 21$ (Sensor 3) to $T = 309$ (Sensor 4), correctly outputting $-1$ for signals that never breached the target increase within the forecast horizon\[cite: 38].



\---



\## 📁 Repository Structure



```text

├── data.csv                # Raw multi-sensor sequential dataset

├── Predict\_T.ipynb         # Model training, evaluation, and T-prediction notebook

├── LSTM\_Prediction.pdf     # Project report documentation

├── .gitignore              # Ignored files and directories

└── README.md               # Project documentation

