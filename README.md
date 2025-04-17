# Amazon Stock Price Forecasting Using LSTM Neural Networks
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.0%2B-red)
![pandas](https://img.shields.io/badge/pandas-1.0%2B-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-yellowgreen)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![license](https://img.shields.io/badge/license-MIT-lightgrey.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen)

This Jupyter notebook demonstrates how to build a Long Short-Term Memory (LSTM) neural network to forecast Amazon stock prices. 
The project uses historical stock price data from AMZN.csv to train an LSTM model for time series forecasting.
___

## Key Features
- Data Preparation: The notebook loads and preprocesses Amazon stock price data, focusing on closing prices over time.
- Feature Engineering: Creates time-lagged features to help the model learn temporal patterns.
- Data Normalization: Uses MinMaxScaler to scale data between -1 and 1 for better model performance.
- LSTM Model: Implements a custom LSTM architecture with PyTorch for sequence prediction.
- Training Pipeline: Includes data loading, model training, and validation processes.
- Time Series Forecasting: Demonstrates how to use the trained model for stock price prediction.
___

## Technical Details
- Layers: 1 LSTM layer with 4 hidden units followed by a fully connected layer
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam with learning rate 0.001
- Batch Size: 16
- Lookback Window: 7 days (using past 7 days to predict next day)
- The `AMZN.csv` dataset is included in the repository
___

## Requirements
- Python 3.7+
- PyTorch
- pandas
- numpy
- matplotlib
- scikit-learn
___

## Installing Python dependencies

I recommend using **Python 3.7 or higher** for this project. If you do not have Python installed in your local environment, please visit [python.org](https://www.python.org/downloads/) for the latest download instruction. 

With Python installed, please go to the Command Line interface of your operating system and use the "pip install" prompts below to install PyTorch, Pandas, NumPy, Matplotlib, scikit-learn respectively. 

- `pip install torch`
- `pip install pandas`
- `pip install numpy`
- `pip install matplotlib`
- `pip install scikit-learn`

___

## Getting the files
-  Use GitHub to clone the repository locally, or download the .zip file of the repository and extract the files.
___

## LSTM Model Definition (models.py)

```python

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM model for stock price forecasting
    Args:
        input_size: Number of input features (1 for univariate time series)
        hidden_size: Number of LSTM hidden units
        num_stacked_layers: Number of stacked LSTM layers
    """
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, 
                           batch_first=True)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Only take the output from the final timestep
        out = self.fc(out[:, -1, :])
        return out
   ```
___

## Data Preprocessing 

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_dataframe_for_lstm(df, n_steps):
    """
    Prepare time series data for LSTM training by creating lag features
    
    Args:
        df: Pandas DataFrame with 'Date' and 'Close' columns
        n_steps: Number of lookback steps (time window size)
    
    Returns:
        shifted_df: DataFrame with lag features
        scaler: Fitted MinMaxScaler object
    """
    df = df.copy()
    df.set_index('Date', inplace=True)

    # Create lag features
    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    
    df.dropna(inplace=True)
    
    # Normalize data between -1 and 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df)
    
    return pd.DataFrame(scaled_data, columns=df.columns), scaler

def create_sequences(data, lookback):
    """
    Create input sequences and targets for LSTM training
    
    Args:
        data: Normalized DataFrame
        lookback: Number of timesteps to look back
    
    Returns:
        X: Input sequences (n_samples, lookback, n_features)
        y: Target values
    """
    X = data[:, 1:]  # All columns except target
    y = data[:, 0]   # Target column (Close price)
    
    # Flip to have most recent timesteps last
    X = np.flip(X, axis=1)
    
    # Reshape for LSTM [samples, timesteps, features]
    X = X.reshape((-1, lookback, 1))
    y = y.reshape((-1, 1))
    
    return X, y
   ```
___

## Long-Term Trend
*Long-term trend prediction of Amazon stock price.*
- Scale: The y-axis ranges from -1.0 to -0.2, representing normalized closing prices.
- Trend: The plot spans ~5000 days (historical data), showing how well the model captures long-term trends.
- Alignment: The predicted values (orange/red line) closely follow the actual closing prices (blue line), indicating the model learns temporal patterns effectively.
- Volatility: Some deviations occur during market fluctuations, suggesting the model may struggle with abrupt changes.

  ![alt text](https://github.com/IlijaMihajlovic/Amazon-Stock-Price-Forecasting-Using-LSTM-Neural-Networks/blob/main/Images/Long-Term%20Trend.png)

___

  ## Short-Term Focus
  *Short-term zoom showing prediction accuracy over 300 days.*
- Scale: The y-axis ranges from 0.0 to 0.8 (normalized), zooming into a 300-day window.
- Precision: Predictions (orange/red) align closely with actual prices (blue), especially in stable periods.
- Lag Effect: Minor delays in predictions during sharp price changes hint at the LSTM’s tendency to "smooth" volatile movements

  ![alt text](https://github.com/IlijaMihajlovic/Amazon-Stock-Price-Forecasting-Using-LSTM-Neural-Networks/blob/main/Images/Short-Term%20Focus%20.png)

## License
```
MIT License

Copyright (c) 2025 Ilija Mihajlovic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```
