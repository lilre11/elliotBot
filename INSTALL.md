# Installation Guide

## Option 1: Minimal Installation (Recommended)

For the fastest and most reliable installation:

```bash
pip install -r requirements-minimal.txt
```

This installs only the essential packages needed for core Elliott Wave functionality.

## Option 2: Full Installation

```bash
pip install -r requirements.txt
```

This includes optional packages like seaborn and testing tools.

## Option 3: Manual Minimal Installation

If you encounter any issues, install packages individually:

```bash
pip install numpy pandas scipy yfinance scikit-learn plotly matplotlib sqlalchemy python-dotenv joblib tqdm requests pyyaml
```

### Step 1: Install Core Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Install Optional ML Libraries (if needed)
```bash
# For TensorFlow (advanced ML models)
pip install tensorflow>=2.13.0

# For PyTorch (alternative ML framework)
pip install torch>=1.11.0

# For XGBoost (gradient boosting)
pip install xgboost>=1.6.0
```

### Step 3: Install TA-Lib (Windows)
TA-Lib can be tricky on Windows. Try these options:

**Option 1: Using pip (may work)**
```bash
pip install TA-Lib
```

**Option 2: Download pre-compiled wheel**
1. Go to https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
2. Download the appropriate `.whl` file for your Python version
3. Install: `pip install path/to/downloaded/file.whl`

**Option 3: Using conda**
```bash
conda install -c conda-forge ta-lib
```

### Step 4: Install Additional TA Libraries (optional)
```bash
pip install ta>=0.10.0
pip install stockstats>=0.5.0
```

## Troubleshooting

### Common Issues

1. **TensorFlow Installation Error**
   - **Solution**: Skip TensorFlow for now, the system works without it
   - **Alternative**: Use scikit-learn models only

2. **TA-Lib Installation Error**
   - **Solution**: The system has built-in technical indicators
   - **Alternative**: Use the custom indicators in `src/data/indicators.py`

3. **Python Version Issues**
   - **Requirement**: Python 3.8 or higher
   - **Check version**: `python --version`

4. **Windows-specific Issues**
   - Install Microsoft Visual C++ Build Tools if needed
   - Use conda instead of pip for problematic packages

### Minimal Installation (No Optional Dependencies)

If you encounter issues, you can run the system with just these packages:

```bash
pip install numpy pandas scipy yfinance plotly matplotlib scikit-learn sqlalchemy python-dotenv joblib tqdm requests pyyaml
```

## Verifying Installation

Run this test script to verify everything works:

```python
# test_installation.py
try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import plotly.graph_objects as go
    from sklearn.ensemble import RandomForestClassifier
    print("✓ Core dependencies installed successfully!")
    
    # Test data loading
    data = yf.download("AAPL", period="1mo", progress=False)
    print("✓ Data loading works!")
    
    # Test basic analysis
    from src.data.data_loader import DataLoader
    from src.analysis.wave_detector import WaveDetector
    
    loader = DataLoader()
    detector = WaveDetector()
    print("✓ Elliott Wave system ready!")
    
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
```

Run with: `python test_installation.py`

## Docker Alternative

If you prefer Docker:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t elliott-bot .
docker run elliott-bot
```

## Environment Setup

Create a `.env` file for API keys:
```env
# Binance API (optional)
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key

# Alpha Vantage API (optional)
ALPHA_VANTAGE_API_KEY=your_api_key
```

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB free space
- **Internet**: Required for data downloads

## Next Steps

After installation:
1. Run `python main.py` to test the system
2. Try examples: `python examples/basic_analysis.py`
3. Read `USAGE_GUIDE.md` for detailed usage instructions
