# Elliott Wave Trading Bot

An advanced AI-powered trading bot that uses **Elliott Wave Theory** to perform technical analysis on financial market data. The system detects impulsive and corrective waves, provides future projections, and includes a comprehensive backtesting engine.

## Features

- **Wave Detection**: Identifies Elliott Wave patterns (1-2-3-4-5 impulse and A-B-C corrective waves)
- **AI/ML Integration**: Uses machine learning models for pattern recognition and wave labeling
- **Fibonacci Analysis**: Incorporates Fibonacci retracement and extension levels
- **Interactive Visualization**: Charts with labeled waves using Plotly
- **Backtesting Engine**: Comprehensive testing framework for strategy validation
- **Modular Architecture**: Scalable design for easy integration with trading APIs
- **Multiple Data Sources**: Support for Yahoo Finance, Binance, and other market data providers

## Project Structure

```
elliottBot/
├── src/
│   ├── data/
│   │   ├── data_loader.py          # OHLCV data loading and preprocessing
│   │   ├── indicators.py           # Technical indicators and ZigZag detection
│   │   └── storage.py              # Data storage and retrieval
│   ├── analysis/
│   │   ├── wave_detector.py        # Elliott Wave detection and labeling
│   │   ├── fibonacci.py            # Fibonacci retracement/extension analysis
│   │   └── pattern_recognition.py  # AI/ML pattern recognition models
│   ├── visualization/
│   │   ├── visualizer.py          # Main charting and visualization
│   │   └── dashboard.py           # Optional web dashboard
│   ├── trading/
│   │   ├── strategy.py            # Signal generation logic
│   │   ├── backtester.py          # Backtesting engine
│   │   └── risk_management.py     # Position sizing and risk controls
│   └── utils/
│       ├── config.py              # Configuration management
│       ├── logger.py              # Logging utilities
│       └── helpers.py             # Common utility functions
├── data/                          # Local data storage
├── models/                        # Trained ML models
├── logs/                          # Application logs
├── tests/                         # Unit tests
├── examples/                      # Usage examples
├── config.yaml                    # Configuration file
├── requirements.txt               # Python dependencies
└── main.py                        # Main application entry point
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd elliottBot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.data.data_loader import DataLoader
from src.analysis.wave_detector import WaveDetector
from src.visualization.visualizer import Visualizer

# Load data
loader = DataLoader()
data = loader.get_yahoo_data("AAPL", period="2y")

# Detect waves
detector = WaveDetector()
waves = detector.detect_waves(data)

# Visualize
viz = Visualizer()
viz.plot_waves(data, waves)
```

## Configuration

Edit `config.yaml` to customize:
- Data sources and API keys
- Wave detection parameters
- Visualization settings
- Backtesting parameters

## Usage Examples

See the `examples/` directory for detailed usage examples including:
- Basic wave detection
- Backtesting strategies
- Custom indicator development
- API integration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results.
