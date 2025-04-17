# Amazon Stock Price Forecasting Using LSTM Neural Networks

![platform-windows](https://img.shields.io/badge/platform-Windows-0078D6.svg)
![platform-macos](https://img.shields.io/badge/platform-macOS-000000.svg)
![python-version](https://img.shields.io/badge/python-3.7%2B-3776AB.svg)
![license](https://img.shields.io/badge/license-MIT-lightgrey.svg)



## Key Features
- Data preprocessing and normalization
- LSTM model architecture
- Training and validation pipeline
- Stock price forecasting
- Visualization of predictions vs actual prices

## Requirements
- Python 3.7+
- PyTorch
- pandas
- numpy
- matplotlib
- scikit-learn

## Getting the files
-  Use GitHub to clone the repository locally, or download the .zip file of the repository and extract the files.


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
