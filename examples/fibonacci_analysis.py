"""
Example: Fibonacci Analysis
This example demonstrates Fibonacci retracement and extension analysis.
"""

import sys
import os
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.data_loader import DataLoader
from src.analysis.fibonacci import FibonacciAnalyzer
from src.visualization.visualizer import WaveVisualizer

warnings.filterwarnings('ignore')


def main():
    print("=== Fibonacci Analysis Example ===")
    
    # Load data
    print("Loading EUR/USD data...")
    loader = DataLoader()
    data = loader.get_yahoo_data("EURUSD=X", period="6mo")
    print(f"Loaded {len(data)} data points")
    
    # Calculate recent swing high and low
    lookback = 50
    recent_data = data.tail(lookback)
    swing_high = recent_data['high'].max()
    swing_low = recent_data['low'].min()
    current_price = data['close'].iloc[-1]
    
    print(f"\nPrice Analysis:")
    print(f"Current Price: {current_price:.4f}")
    print(f"Recent Swing High: {swing_high:.4f}")
    print(f"Recent Swing Low: {swing_low:.4f}")
    
    # Perform Fibonacci analysis
    print("\nPerforming Fibonacci analysis...")
    fib_analyzer = FibonacciAnalyzer()
    
    # Analyze for uptrend (retracement from high)
    fib_analysis = fib_analyzer.analyze_retracement(
        swing_high=swing_high,
        swing_low=swing_low,
        current_price=current_price,
        trend_direction='up'
    )
    
    print(f"\nFibonacci Retracement Levels:")
    for level in fib_analysis.retracements:
        if level.is_key_level:
            distance_pct = abs(current_price - level.price) / current_price * 100
            print(f"  {level.ratio:.1%}: {level.price:.4f} (Distance: {distance_pct:.2f}%)")
    
    print(f"\nFibonacci Extension Levels:")
    for level in fib_analysis.extensions:
        if level.is_key_level:
            print(f"  {level.ratio:.1%}: {level.price:.4f}")
    
    # Analyze current price position
    price_analysis = fib_analyzer.analyze_price_at_fibonacci(current_price, fib_analysis)
    print(f"\nPrice Position Analysis:")
    print(f"Position: {price_analysis['price_position']}")
    print(f"At Fibonacci Level: {price_analysis['at_fibonacci_level']}")
    
    if price_analysis['next_support']:
        support = price_analysis['next_support']
        print(f"Next Support: {support.price:.4f} ({support.ratio:.1%})")
    
    if price_analysis['next_resistance']:
        resistance = price_analysis['next_resistance']
        print(f"Next Resistance: {resistance.price:.4f} ({resistance.ratio:.1%})")
    
    # Calculate wave targets for different scenarios
    print(f"\nWave Target Examples:")
    
    # Example: Wave 3 targets
    targets_wave3 = fib_analyzer.calculate_wave_targets(
        wave_start=swing_low,
        wave_end=current_price,
        wave_type='3',
        reference_wave_start=swing_low,
        reference_wave_end=swing_high
    )
    
    if targets_wave3:
        print("Wave 3 Targets:")
        for target_name, target_price in targets_wave3.items():
            print(f"  {target_name}: {target_price:.4f}")
    
    # Find confluence zones
    print(f"\nFinding Fibonacci confluence zones...")
    confluence_zones = fib_analyzer.get_fibonacci_confluence(data, lookback_periods=100)
    
    if confluence_zones:
        print("Confluence Zones:")
        for i, zone in enumerate(confluence_zones):
            print(f"  Zone {i+1}: {zone['price']:.4f} (Strength: {zone['strength']}, "
                  f"Ratios: {[f'{r:.1%}' for r in zone['ratios']]})")
    
    # Create visualization
    print("\nCreating Fibonacci chart...")
    visualizer = WaveVisualizer()
    fig = visualizer.plot_fibonacci_analysis(
        data, 
        fib_analysis, 
        title="EUR/USD Fibonacci Analysis"
    )
    
    # Show the chart
    print("Displaying chart...")
    fig.show()
    
    print("Fibonacci analysis example completed!")


if __name__ == "__main__":
    main()
