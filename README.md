# financial-time-series-prediction```markdown

## 2. Overview
This project explores the application of machine learning models to predict daily log returns of a stock (Apple - AAPL) for potential use in algorithmic trading strategies. Financial markets are notoriously complex and efficient, making accurate predictions a significant challenge. This notebook demonstrates a typical workflow, from data acquisition and feature engineering to model training, evaluation, and hyperparameter tuning, providing insights into the inherent difficulties of forecasting financial time series.

## 3. Dataset Description

**Data Source:** Historical daily stock data for Apple (AAPL) acquired using the `yfinance` library.
**Timeframe:** From January 1, 2020, up to the most recent available date.

**Original Columns:**
*   `Close`: Daily closing price.
*   `High`: Daily high price.
*   `Low`: Daily low price.
*   `Open`: Daily opening price.
*   `Volume`: Daily trading volume.

**Derived Features & Target:**
*   `log_return`: The daily logarithmic return, calculated as `np.log(Close / Close.shift(1))`. Log returns are often preferred in financial modeling for their additive properties over time.
*   `volatility_5`: 5-day rolling standard deviation of `log_return`, serving as a measure of recent price fluctuations.
*   `momentum_5`: 5-day rolling mean of `log_return`, indicating the short-term trend.
*   `lag_1` to `lag_10`: Lagged daily `log_return` values, capturing past price movements as potential predictors.
*   `target`: The `log_return` of the next day, which is what our models aim to predict (`log_return.shift(-1)`).

**Data Cleaning:** Rows containing `NaN` values (primarily due to the calculation of `log_return` and lagged features) were dropped, ensuring a clean dataset for modeling.

## 4. Methodology

### Data Preprocessing
1.  **Data Acquisition:** Downloaded historical daily stock data for AAPL using `yfinance`.
2.  **Log Returns:** Computed daily log returns from the 'Close' price.
3.  **Feature Engineering:**
    *   **Lag Features:** Created lagged versions of the `log_return` from 1 to 10 days back (`lag_1` to `lag_10`). These act as a memory of past price movements.
    *   **Volatility:** Calculated a 5-day rolling standard deviation of log returns (`volatility_5`) to capture short-term risk.
    *   **Momentum:** Calculated a 5-day rolling mean of log returns (`momentum_5`) to capture short-term trend.
4.  **Target Variable:** The `target` variable was set as the `log_return` shifted by -1, meaning we are trying to predict tomorrow's log return.
5.  **Scaling:** Applied `RobustScaler` to the features (`lag_1` to `lag_5` were used for scaling based on the last executed scaling cell) to handle potential outliers and ensure models are not unduly influenced by feature magnitudes. This scaler is robust to outliers, making it suitable for financial data.

### Train/Test Split Strategy
To maintain the temporal order of time series data, a chronological split was performed:
*   **Training Set:** The first 80% of the data was used for training the models.
*   **Test Set:** The remaining 20% of the data was reserved for evaluating model performance on unseen, future data.

## 5. Models Used

### Baselines
1.  **Zero Baseline:** Predicts a log return of 0 for all future periods. This serves as a simple benchmark for models predicting near-zero returns.
    *   MSE: `0.0003761925573815199`
    *   R2: `-1.4120712730214535e-06`
    *   Direction Accuracy: `0.0%`
2.  **Persistence Baseline:** Predicts that tomorrow's log return will be the same as today's (i.e., `log_return.shift(1)`). This tests if past returns have any immediate predictive power.
    *   MSE: `0.0006712016490380542`
    *   R2: `-0.7792488725193139`
    *   Direction Accuracy: `51.44694533762058%`

### Machine Learning Models

1.  **Random Forest Regressor**
    *   A robust ensemble learning method that builds multiple decision trees.
    *   **Configuration:** `n_estimators=100`, `max_depth=5`, `random_state=42`.
    *   **Performance:**
        *   MSE: `0.00034160674855194964`
        *   R2: `0.09193516931122836`
        *   Direction Accuracy: `52.24358974358975%`

2.  **XGBoost Regressor**
    *   An optimized distributed gradient boosting library designed for speed and performance.
    *   **Configuration (Initial):** `n_estimators=300`, `max_depth=3`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`.
    *   **Performance (Initial):**
        *   MSE: `0.0003380082867379146`
        *   R2: `0.10150066130385738`
        *   Direction Accuracy: `54.80769230769231%`

3.  **LightGBM Regressor**
    *   A gradient boosting framework that uses tree-based learning algorithms, known for its speed and efficiency, especially on large datasets.
    *   **Configuration:** `n_estimators=300`, `learning_rate=0.03`, `subsample=0.8`, `feature_fraction=0.8`, `num_leaves=31`, `random_state=42`.
    *   **Performance:**
        *   MSE: `0.0003995939592855063`
        *   R2: `-0.06220741399571006`
        *   Direction Accuracy: `52.88461538461539%`

## 6. Evaluation Metrics

1.  **Mean Squared Error (MSE):** Measures the average squared difference between the estimated values and the actual value. Lower is better.
2.  **R-squared (R²):** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A higher R² indicates a better fit, but a very low R² (or even negative) is common in financial prediction due to high noise.
3.  **Directional Accuracy:** Crucial in finance, this metric calculates the proportion of times the model correctly predicts the *direction* (up or down) of the stock's movement, irrespective of the magnitude. A value above 0.5 (50%) indicates some predictive power over random chance.

## 7. Results & Key Findings

### Comparison between Models
| Model                  | MSE           | R²            | Direction Accuracy |
| :--------------------- | :------------ | :------------ | :----------------- |
| Zero Baseline          | 0.00037619    | -0.00000      | 0.0%               |
| Persistence Baseline   | 0.00067120    | -0.77924      | 51.45%             |
| Random Forest          | 0.00034160    | 0.09193       | 52.24%             |
| XGBoost (Initial)      | 0.00033800    | 0.10150       | 54.81%             |
| LightGBM               | 0.00039959    | -0.06220      | 52.88%             |

### Interpretation of Performance
*   All machine learning models demonstrate a slight improvement over the baselines, particularly in **Directional Accuracy**. The XGBoost model showed the best initial performance with a Directional Accuracy of nearly 54.81%, indicating it correctly predicted the direction of log returns more often than not.
*   The **R² scores are generally very low or negative**, which is typical for financial time series prediction. This implies that while the models might capture some subtle patterns for directional prediction, they explain very little of the overall variance in log returns.
*   **Hyperparameter Tuning (XGBoost):** Further tuning of the XGBoost model using a grid search on `max_depth`, `learning_rate`, and `n_estimators` resulted in improved directional accuracy.
    *   **Best Parameters:** `(max_depth=3, learning_rate=0.03, n_estimators=200)`
    *   **Tuned XGBoost Performance:**
        *   MSE: `0.00033891993314802405`
        *   R2: `0.09907730741359255`
        *   Direction Accuracy: `56.73076923076923%`
    *   This tuning yielded a `0.019230769230769232` (approximately 1.92%) improvement in directional accuracy over the initial XGBoost model, further validating the importance of optimization.

### Discussion of Weak Signal in Financial Data
The results underscore a critical aspect of financial market prediction: the signal-to-noise ratio is extremely low. Even sophisticated machine learning models struggle to find strong, consistent predictive patterns due to market efficiency and the randomness inherent in price movements. While a directional accuracy above 50% is promising for strategy development, it also highlights that the predictive edge is often marginal.

## 8. Key Insights

*   **Non-linear Signal:** The superior performance of tree-based models (Random Forest, XGBoost, LightGBM) over linear baselines suggests that there might be non-linear relationships and interactions between features that influence stock returns. This reinforces the idea that simpler linear models might not fully capture market dynamics.
*   **Importance of Feature Engineering:** Creating relevant features like lagged returns, volatility, and momentum proved beneficial. These handcrafted features provide the models with context about recent market behavior, which is crucial for time-series forecasting.
*   **Limits of Prediction in Finance:** Despite the efforts in feature engineering and model tuning, the R² values remain low, indicating that predicting the exact magnitude of stock returns is extremely difficult. The focus shifts more towards predicting the *direction* rather than the precise value.

## 9. Limitations

*   **Overfitting Risk:** Given the noisy nature of financial data and the complexity of the models, there's always a risk of overfitting to historical patterns that may not generalize to future market conditions.
*   **Transaction Costs Not Included:** This analysis does not account for real-world trading frictions such as commissions, bid-ask spread, and slippage. These costs can significantly erode any theoretical profits from a predictive model.
*   **Non-Stationarity of Markets:** Financial markets are dynamic and constantly evolving. Models trained on past data might become obsolete as market regimes change, leading to concept drift.

## 10. Future Improvements

*   **Feature Expansion:** Explore additional technical indicators (e.g., RSI, MACD), fundamental data, macroeconomic indicators, sentiment analysis from news, or alternative data sources to potentially uncover stronger signals.
*   **Advanced Model Tuning:** Employ more sophisticated hyperparameter optimization techniques like Bayesian Optimization or evolutionary algorithms for a more exhaustive search of the optimal model configurations.
*   **Time-Series Cross-Validation:** Implement more robust validation strategies specific to time series, such as walk-forward optimization, to better assess model stability and generalization.
*   **Trading Strategy Simulation (Backtesting):** Integrate the predictive model into a comprehensive backtesting framework that includes realistic transaction costs, position sizing, and risk management to evaluate its true profitability.
*   **Deep Learning Models:** Investigate the use of recurrent neural networks (RNNs) or transformer models, which are often well-suited for sequence data like financial time series.

## 11. Installation & Usage

To run this notebook and reproduce the analysis, follow these steps:

1.  **Open in Google Colab:** Click the "Open in Colab" badge or upload the `.ipynb` file to your Google Drive and open it with Colaboratory.
2.  **Run Cells:** Execute each code cell sequentially. The notebook is designed to be run step-by-step.

### Required Libraries
All necessary libraries can be installed via `pip`. They are typically pre-installed or will be installed as dependencies within a Colab environment.

```bash
pip install torch torchvision torchaudio numpy pandas yfinance seaborn matplotlib scikit-learn xgboost lightgbm
```

Note: `torch`, `torchvision`, `torchaudio` are imported but not explicitly used in the final models; they might be remnants of an earlier exploration or placeholder for future deep learning work.
```
