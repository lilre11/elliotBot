"""
Common utility functions for Elliott Wave Trading Bot.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import json
from datetime import datetime, timedelta
import pickle
import os
from pathlib import Path


def normalize_price_data(data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize price data for machine learning.
    
    Args:
        data: OHLCV DataFrame
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Normalized DataFrame
    """
    result = data.copy()
    price_columns = ['open', 'high', 'low', 'close']
    
    if method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        result[price_columns] = scaler.fit_transform(result[price_columns])
    
    elif method == 'zscore':
        for col in price_columns:
            result[col] = (result[col] - result[col].mean()) / result[col].std()
    
    elif method == 'robust':
        for col in price_columns:
            median = result[col].median()
            mad = np.median(np.abs(result[col] - median))
            result[col] = (result[col] - median) / mad
    
    return result


def calculate_fibonacci_levels(high: float, low: float, trend: str = 'uptrend') -> Dict[str, float]:
    """
    Calculate Fibonacci retracement levels.
    
    Args:
        high: High price
        low: Low price
        trend: 'uptrend' or 'downtrend'
        
    Returns:
        Dictionary of Fibonacci levels
    """
    fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    levels = {}
    
    if trend == 'uptrend':
        # Retracement levels from high to low
        for ratio in fib_ratios:
            levels[f"fib_{ratio}"] = high - (high - low) * ratio
    else:
        # Extension levels from low to high
        for ratio in fib_ratios:
            levels[f"fib_{ratio}"] = low + (high - low) * ratio
    
    return levels


def calculate_fibonacci_extensions(
    swing_low: float, 
    swing_high: float, 
    current_low: float
) -> Dict[str, float]:
    """
    Calculate Fibonacci extension levels.
    
    Args:
        swing_low: Previous swing low
        swing_high: Previous swing high
        current_low: Current retracement low
        
    Returns:
        Dictionary of extension levels
    """
    wave_height = swing_high - swing_low
    extension_ratios = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618]
    
    extensions = {}
    for ratio in extension_ratios:
        extensions[f"ext_{ratio}"] = current_low + wave_height * ratio
    
    return extensions


def find_peaks_and_troughs(
    data: pd.Series, 
    window: int = 5, 
    min_distance: int = 1
) -> Tuple[List[int], List[int]]:
    """
    Find peaks and troughs in a price series.
    
    Args:
        data: Price series
        window: Window size for peak detection
        min_distance: Minimum distance between peaks
        
    Returns:
        Tuple of (peak_indices, trough_indices)
    """
    from scipy.signal import find_peaks
    
    # Find peaks (local maxima)
    peaks, _ = find_peaks(data.values, distance=min_distance)
    
    # Find troughs (local minima)
    troughs, _ = find_peaks(-data.values, distance=min_distance)
    
    return peaks.tolist(), troughs.tolist()


def calculate_price_velocity(data: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Calculate price velocity (rate of change).
    
    Args:
        data: OHLCV DataFrame
        window: Window for velocity calculation
        
    Returns:
        Price velocity series
    """
    price_change = data['close'].diff(window)
    time_change = window  # Assuming constant time intervals
    velocity = price_change / time_change
    
    return velocity


def calculate_wave_strength(
    start_price: float, 
    end_price: float, 
    volume_data: pd.Series,
    time_duration: int
) -> float:
    """
    Calculate wave strength based on price move, volume, and time.
    
    Args:
        start_price: Wave start price
        end_price: Wave end price
        volume_data: Volume data for the wave period
        time_duration: Duration of the wave in periods
        
    Returns:
        Wave strength score
    """
    # Price component
    price_change = abs(end_price - start_price) / start_price
    
    # Volume component (normalized)
    avg_volume = volume_data.mean()
    volume_strength = avg_volume / volume_data.rolling(20).mean().mean()
    
    # Time component (shorter waves generally stronger)
    time_factor = 1.0 / max(time_duration, 1)
    
    # Combine factors
    strength = price_change * volume_strength * time_factor
    
    return min(strength, 1.0)  # Cap at 1.0


def calculate_support_resistance(
    data: pd.DataFrame, 
    window: int = 20,
    min_touches: int = 2
) -> Dict[str, List[float]]:
    """
    Identify support and resistance levels.
    
    Args:
        data: OHLCV DataFrame
        window: Window for level identification
        min_touches: Minimum touches to confirm level
        
    Returns:
        Dictionary with support and resistance levels
    """
    levels = {'support': [], 'resistance': []}
    
    # Rolling highs and lows
    rolling_high = data['high'].rolling(window).max()
    rolling_low = data['low'].rolling(window).min()
    
    # Find potential resistance levels
    resistance_candidates = data[data['high'] == rolling_high]['high'].unique()
    
    # Find potential support levels
    support_candidates = data[data['low'] == rolling_low]['low'].unique()
    
    # Count touches for each level
    tolerance = 0.01  # 1% tolerance
    
    for level in resistance_candidates:
        touches = len(data[abs(data['high'] - level) / level <= tolerance])
        if touches >= min_touches:
            levels['resistance'].append(level)
    
    for level in support_candidates:
        touches = len(data[abs(data['low'] - level) / level <= tolerance])
        if touches >= min_touches:
            levels['support'].append(level)
    
    return levels


def calculate_trend_strength(data: pd.DataFrame, period: int = 20) -> float:
    """
    Calculate trend strength using various indicators.
    
    Args:
        data: OHLCV DataFrame
        period: Period for calculation
        
    Returns:
        Trend strength (-1 to 1, negative for downtrend)
    """
    # Price momentum
    price_momentum = (data['close'].iloc[-1] - data['close'].iloc[-period]) / data['close'].iloc[-period]
    
    # Moving average slope
    ma = data['close'].rolling(period).mean()
    ma_slope = (ma.iloc[-1] - ma.iloc[-period]) / period
    
    # Volume trend
    volume_trend = data['volume'].rolling(period).mean().iloc[-1] / data['volume'].rolling(period*2).mean().iloc[-1]
    
    # Combine indicators
    trend_strength = (price_momentum + ma_slope * 10 + (volume_trend - 1)) / 3
    
    return np.clip(trend_strength, -1, 1)


def save_model(model: Any, filepath: str, metadata: Dict[str, Any] = None):
    """
    Save a machine learning model with metadata.
    
    Args:
        model: Model object to save
        filepath: Path to save the model
        metadata: Additional metadata to save
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'metadata': metadata or {},
        'saved_at': datetime.now().isoformat(),
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)


def load_model(filepath: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a machine learning model with metadata.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Tuple of (model, metadata)
    """
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data['model'], model_data.get('metadata', {})


def validate_ohlcv_data(data: pd.DataFrame) -> List[str]:
    """
    Validate OHLCV data integrity.
    
    Args:
        data: OHLCV DataFrame
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Check required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    if errors:
        return errors
    
    # Check for negative values
    if (data[['open', 'high', 'low', 'close']] <= 0).any().any():
        errors.append("Found non-positive price values")
    
    if (data['volume'] < 0).any():
        errors.append("Found negative volume values")
    
    # Check OHLC relationships
    if (data['high'] < data['low']).any():
        errors.append("Found high < low")
    
    if (data['high'] < data['open']).any():
        errors.append("Found high < open")
    
    if (data['high'] < data['close']).any():
        errors.append("Found high < close")
    
    if (data['low'] > data['open']).any():
        errors.append("Found low > open")
    
    if (data['low'] > data['close']).any():
        errors.append("Found low > close")
    
    # Check for missing data
    if data.isnull().any().any():
        errors.append("Found missing values")
    
    return errors


def resample_data(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to different timeframe.
    
    Args:
        data: OHLCV DataFrame
        timeframe: Target timeframe ('1H', '4H', '1D', '1W', etc.)
        
    Returns:
        Resampled DataFrame
    """
    resampled = data.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled


def calculate_correlation_matrix(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate correlation matrix between multiple instruments.
    
    Args:
        data_dict: Dictionary of symbol -> DataFrame
        
    Returns:
        Correlation matrix
    """
    closes = pd.DataFrame()
    
    for symbol, data in data_dict.items():
        closes[symbol] = data['close']
    
    return closes.corr()


def export_data(data: pd.DataFrame, filepath: str, format: str = 'csv'):
    """
    Export data to various formats.
    
    Args:
        data: DataFrame to export
        filepath: Output file path
        format: Export format ('csv', 'json', 'parquet')
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        data.to_csv(filepath)
    elif format == 'json':
        data.to_json(filepath, orient='index', date_format='iso')
    elif format == 'parquet':
        data.to_parquet(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def import_data(filepath: str, format: str = None) -> pd.DataFrame:
    """
    Import data from various formats.
    
    Args:
        filepath: Input file path
        format: File format (auto-detected if None)
        
    Returns:
        Imported DataFrame
    """
    if format is None:
        format = Path(filepath).suffix.lower().lstrip('.')
    
    if format == 'csv':
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
    elif format == 'json':
        return pd.read_json(filepath, orient='index')
    elif format == 'parquet':
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def create_feature_matrix(
    data: pd.DataFrame, 
    features: List[str], 
    lookback: int = 20
) -> np.ndarray:
    """
    Create feature matrix for machine learning.
    
    Args:
        data: DataFrame with features
        features: List of feature column names
        lookback: Number of lookback periods
        
    Returns:
        Feature matrix
    """
    feature_matrix = []
    
    for i in range(lookback, len(data)):
        row_features = []
        for feature in features:
            # Add current value and historical values
            current_val = data[feature].iloc[i]
            historical_vals = data[feature].iloc[i-lookback:i].values
            
            row_features.extend([current_val])
            row_features.extend(historical_vals)
        
        feature_matrix.append(row_features)
    
    return np.array(feature_matrix)


if __name__ == "__main__":
    # Example usage
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Validate data
    errors = validate_ohlcv_data(sample_data)
    print(f"Validation errors: {errors}")
    
    # Calculate Fibonacci levels
    fib_levels = calculate_fibonacci_levels(110, 90, 'uptrend')
    print(f"Fibonacci levels: {fib_levels}")
    
    # Calculate trend strength
    trend = calculate_trend_strength(sample_data)
    print(f"Trend strength: {trend}")
    
    # Find support/resistance
    levels = calculate_support_resistance(sample_data)
    print(f"Support/Resistance levels: {levels}")
