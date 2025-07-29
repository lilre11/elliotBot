"""
Test Installation
Quick test to verify that Elliott Wave Bot installation works correctly.
"""

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    try:
        import numpy as np
        print("✓ NumPy")
    except ImportError:
        print("✗ NumPy - REQUIRED")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas")
    except ImportError:
        print("✗ Pandas - REQUIRED")
        return False
    
    try:
        import scipy
        print("✓ SciPy")
    except ImportError:
        print("✗ SciPy - REQUIRED")
        return False
    
    try:
        import yfinance as yf
        print("✓ yfinance")
    except ImportError:
        print("✗ yfinance - REQUIRED")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError:
        print("✗ scikit-learn - REQUIRED")
        return False
    
    try:
        import plotly
        print("✓ Plotly")
    except ImportError:
        print("✗ Plotly - REQUIRED")
        return False
    
    try:
        import matplotlib
        print("✓ Matplotlib")
    except ImportError:
        print("✗ Matplotlib - REQUIRED")
        return False
    
    try:
        import yaml
        print("✓ PyYAML")
    except ImportError:
        print("✗ PyYAML - REQUIRED")
        return False
    
    # Optional packages
    try:
        import seaborn
        print("✓ Seaborn (optional)")
    except ImportError:
        print("○ Seaborn (optional) - not installed")
    
    try:
        import ccxt  # type: ignore[import-untyped]
        print("✓ CCXT (optional)")
    except ImportError:
        print("○ CCXT (optional) - not installed")
    
    try:
        import talib  # type: ignore[import-untyped]
        print("✓ TA-Lib (optional)")
    except ImportError:
        print("○ TA-Lib (optional) - not installed")

    return True


def test_data_loading():
    """Test that data can be loaded from Yahoo Finance."""
    print("\nTesting data loading...")
    
    try:
        import yfinance as yf
        
        # Test downloading a small amount of data
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="5d")
        
        if len(data) > 0:
            print("✓ Yahoo Finance data loading works")
            print(f"  Downloaded {len(data)} days of AAPL data")
            return True
        else:
            print("✗ No data received from Yahoo Finance")
            return False
            
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return False


def test_basic_analysis():
    """Test basic Elliott Wave analysis."""
    print("\nTesting Elliott Wave components...")
    
    try:
        import sys
        import os
        
        # Add src to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        from src.data.data_loader import DataLoader
        print("✓ DataLoader import")
        
        from src.analysis.wave_detector import WaveDetector
        print("✓ WaveDetector import")
        
        from src.analysis.fibonacci import FibonacciAnalyzer
        print("✓ FibonacciAnalyzer import")
        
        from src.visualization.visualizer import WaveVisualizer
        print("✓ WaveVisualizer import")
        
        from src.trading.strategy import ElliottWaveStrategy
        print("✓ ElliottWaveStrategy import")
        
        # Test basic instantiation
        loader = DataLoader()
        detector = WaveDetector()
        analyzer = FibonacciAnalyzer()
        
        print("✓ All components can be instantiated")
        return True
        
    except Exception as e:
        print(f"✗ Elliott Wave component error: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        import os
        import yaml
        
        config_path = "config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("✓ Configuration file loaded")
            return True
        else:
            print("○ config.yaml not found - will use defaults")
            return True
            
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("ELLIOTT WAVE BOT INSTALLATION TEST")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test data loading
    if not test_data_loading():
        all_passed = False
    
    # Test Elliott Wave components
    if not test_basic_analysis():
        all_passed = False
    
    # Test configuration
    if not test_config():
        all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED! Elliott Wave Bot is ready to use.")
        print("\nNext steps:")
        print("1. Run: python main.py")
        print("2. Try: python examples/basic_analysis.py")
        print("3. Read: USAGE_GUIDE.md")
    else:
        print("❌ SOME TESTS FAILED. Please check the installation.")
        print("\nTroubleshooting:")
        print("1. Try: pip install -r requirements-minimal.txt")
        print("2. Check Python version: python --version (needs 3.8+)")
        print("3. See INSTALL.md for detailed instructions")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
