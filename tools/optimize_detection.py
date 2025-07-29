"""
Elliott Wave Configuration Tuner
This script helps optimize wave detection parameters for better results.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import DataLoader
from src.analysis.wave_detector import WaveDetector
import yaml

def test_wave_detection_parameters():
    """Test different parameters to optimize wave detection"""
    
    print("🔧 Elliott Wave Detection Parameter Tuner")
    print("=" * 50)
    
    # Load test data
    loader = DataLoader()
    data = loader.get_yahoo_data('AAPL', period='1y')
    
    # Test different threshold values
    thresholds = [0.03, 0.04, 0.05, 0.06, 0.07]
    
    print(f"Testing wave detection on AAPL (1y data - {len(data)} points)")
    print("\nThreshold | Waves Detected | Avg Confidence")
    print("-" * 45)
    
    best_threshold = 0.05
    best_score = 0
    
    for threshold in thresholds:
        # Create detector with specific threshold
        detector = WaveDetector()
        detector.zigzag_threshold = threshold
        
        waves = detector.detect_waves(data)
        avg_confidence = sum(w.confidence for w in waves) / len(waves) if waves else 0
        
        # Simple scoring: balance between number of waves and confidence
        score = len(waves) * avg_confidence if waves else 0
        
        print(f"  {threshold:.2f}    |      {len(waves):2d}        |    {avg_confidence:.3f}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    print("-" * 45)
    print(f"✅ Recommended threshold: {best_threshold:.2f}")
    
    return best_threshold

def update_config_with_optimal_settings(threshold):
    """Update the configuration file with optimal settings"""
    
    config_path = 'config.yaml'
    
    # Read existing config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {}
    
    # Update wave detection settings
    if 'wave_detection' not in config:
        config['wave_detection'] = {}
    
    config['wave_detection']['zigzag_threshold'] = threshold
    config['wave_detection']['min_wave_length'] = 3  # Reduced for more sensitivity
    config['wave_detection']['confidence_threshold'] = 0.6  # Lowered for more waves
    
    # Write updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Updated {config_path} with optimal settings")

def main():
    print("🎯 Elliott Wave Bot Optimization")
    print("=" * 40)
    
    # Test and find optimal parameters
    optimal_threshold = test_wave_detection_parameters()
    
    # Update configuration
    update_config_with_optimal_settings(optimal_threshold)
    
    print("\n🚀 Elliott Wave Bot has been optimized!")
    print("Run your analysis again to see improved wave detection.")

if __name__ == "__main__":
    main()
