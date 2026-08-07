# Finance Pipeline

This directory handles the retrieval and alignment of market data for the 26 selected firms in the dataset.

## Scripts

- **`vix_pipeline.py`**: A script to pull required financial market data, primarily using the `yfinance` library.

## Data Collected

1. **Firm-level Returns**: Daily closing prices are fetched and converted into daily log returns ($`\ln P_{i,t} - \ln P_{i,t-1}`$).
2. **S&P 500 Index**: Acts as a control for broad market movements.
3. **VIX Volatility Index**: Acts as a control for market-wide volatility swings, especially pertinent to the early 2020 pandemic shock.

These financial metrics are subsequently aligned with the daily average stance score of each firm, allowing the downstream Panel VAR models to account for endogenous and common market effects when assessing the predictive power of media stance on stock returns.
