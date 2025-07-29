"""
Example: Basic Elliott Wave Analysis
This example demonstrates how to perform basic Elliott Wave analysis on a stock.
"""

import sys
import os
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Add project root to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import DataLoader
from src.analysis.wave_detector import WaveDetector
from src.visualization.visualizer import WaveVisualizer

warnings.filterwarnings('ignore')


def main():
    print("=== Basic Elliott Wave Analysis Example ===")
    
    # Load data for Apple Inc.
    print("Loading AAPL data...")
    loader = DataLoader()
    data = loader.get_yahoo_data("AAPL", period="1y")
    print(f"Loaded {len(data)} data points")
    
    # Detect Elliott Waves
    print("Detecting Elliott Waves...")
    detector = WaveDetector()
    waves = detector.detect_waves(data)
    print(f"Detected {len(waves)} waves")
    
    # Print wave information
    if waves:
        print("\nDetected Waves:")
        for i, wave in enumerate(waves):
            print(f"Wave {i+1}: {wave.wave_type.value} "
                  f"({wave.start_point.timestamp.strftime('%Y-%m-%d')} -> "
                  f"{wave.end_point.timestamp.strftime('%Y-%m-%d')}) "
                  f"Confidence: {wave.confidence:.2f}")
    
    # Get current wave count
    current_analysis = detector.get_current_wave_count(data)
    print(f"\nCurrent Wave Analysis:")
    print(f"Current Wave: {current_analysis['current_wave']}")
    print(f"Confidence: {current_analysis['confidence']:.2f}")
    print(f"Analysis: {current_analysis['analysis']}")
    
    # Create visualization
    print("\nCreating visualization...")
    visualizer = WaveVisualizer()
    fig = visualizer.plot_waves(data, waves, title="AAPL Elliott Wave Analysis")
    
    # Save chart as HTML file
    print("Saving chart as HTML...")
    fig.write_html("elliott_wave_analysis.html")
    print("Chart saved as 'elliott_wave_analysis.html' - you can open this file in your browser")
    
    print("Example completed!")


if __name__ == "__main__":
    main()
