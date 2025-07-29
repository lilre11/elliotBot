"""
Elliott Wave Bot Health Check & Error Fixing System
This script diagnoses and fixes common issues in the Elliott Wave Bot.
"""

import sys
import os
import warnings
import traceback
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking Dependencies...")
    
    required_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        ('plotly', 'plotly'),
        ('yfinance', 'yfinance'),
        ('yaml', 'yaml'),
        ('scipy', 'scipy')
    ]
    
    optional_packages = [
        ('ccxt', 'ccxt'),
        ('talib', 'talib'),
        ('tensorflow', 'tensorflow'),
        ('torch', 'torch')
    ]
    
    missing_required = []
    missing_optional = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name} - REQUIRED")
            missing_required.append(package_name)
    
    for package_name, import_name in optional_packages:
        try:
            __import__(import_name)
            print(f"  ✅ {package_name} (optional)")
        except ImportError:
            print(f"  ⚠️  {package_name} (optional - not installed)")
            missing_optional.append(package_name)
    
    return missing_required, missing_optional

def check_data_loading():
    """Test data loading functionality"""
    print("\n📊 Testing Data Loading...")
    
    try:
        from src.data.data_loader import DataLoader
        loader = DataLoader()
        
        # Test Yahoo Finance
        data = loader.get_yahoo_data('AAPL', period='1mo')
        if len(data) > 0:
            print(f"  ✅ Yahoo Finance: Loaded {len(data)} records")
        else:
            print("  ❌ Yahoo Finance: No data loaded")
            return False
            
        # Check data structure
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_columns if col not in data.columns]
        
        if missing_cols:
            print(f"  ❌ Missing columns: {missing_cols}")
            return False
        else:
            print("  ✅ Data structure is correct")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Data loading error: {str(e)}")
        return False

def check_wave_detection():
    """Test wave detection functionality"""
    print("\n🌊 Testing Wave Detection...")
    
    try:
        from src.data.data_loader import DataLoader
        from src.analysis.wave_detector import WaveDetector
        
        loader = DataLoader()
        detector = WaveDetector()
        
        # Load test data
        data = loader.get_yahoo_data('AAPL', period='6mo')
        
        # Detect waves
        waves = detector.detect_waves(data)
        
        print(f"  ✅ Wave detection works: {len(waves)} waves detected")
        
        if waves:
            for i, wave in enumerate(waves[:3]):  # Show first 3 waves
                wave_label = wave.wave_type.value.split('_')[-1] if '_' in wave.wave_type.value else wave.wave_type.value
                print(f"    Wave {i+1}: {wave_label} (confidence: {wave.confidence:.2f})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Wave detection error: {str(e)}")
        traceback.print_exc()
        return False

def check_visualization():
    """Test visualization functionality"""
    print("\n🎨 Testing Visualization...")
    
    try:
        from src.data.data_loader import DataLoader
        from src.analysis.wave_detector import WaveDetector
        from src.visualization.visualizer import WaveVisualizer
        
        loader = DataLoader()
        detector = WaveDetector()
        visualizer = WaveVisualizer()
        
        # Load test data
        data = loader.get_yahoo_data('AAPL', period='3mo')
        waves = detector.detect_waves(data)
        
        # Create visualization
        fig = visualizer.plot_waves(data, waves, title="Health Check Test")
        
        # Test saving
        test_filename = "health_check_test.html"
        fig.write_html(test_filename)
        
        # Check if file was created
        if os.path.exists(test_filename):
            print(f"  ✅ Visualization works: {test_filename} created")
            os.remove(test_filename)  # Clean up
            return True
        else:
            print("  ❌ Visualization: File not created")
            return False
            
    except Exception as e:
        print(f"  ❌ Visualization error: {str(e)}")
        return False

def check_fibonacci_analysis():
    """Test Fibonacci analysis functionality"""
    print("\n📐 Testing Fibonacci Analysis...")
    
    try:
        from src.analysis.fibonacci import FibonacciAnalyzer
        
        analyzer = FibonacciAnalyzer()
        
        # Test basic functionality
        analysis = analyzer.analyze_retracement(100.0, 50.0, 75.0, 'UP')
        
        if analysis and hasattr(analysis, 'retracement_levels'):
            print(f"  ✅ Fibonacci analysis works: {len(analysis.retracement_levels)} levels calculated")
            return True
        else:
            print("  ❌ Fibonacci analysis: Invalid result")
            return False
            
    except Exception as e:
        print(f"  ❌ Fibonacci analysis error: {str(e)}")
        return False

def check_configuration():
    """Check configuration file and settings"""
    print("\n⚙️  Checking Configuration...")
    
    config_files = ['config.yaml', 'src/config.yaml']
    config_found = False
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"  ✅ Found config file: {config_file}")
            config_found = True
            break
    
    if not config_found:
        print("  ⚠️  No config file found - using defaults")
    
    return True

def fix_common_issues():
    """Attempt to fix common issues automatically"""
    print("\n🔧 Attempting to Fix Common Issues...")
    
    fixes_applied = []
    
    # Fix 1: Create missing directories
    directories = ['data', 'logs', 'output', 'charts']
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                fixes_applied.append(f"Created directory: {directory}")
            except Exception as e:
                print(f"  ❌ Failed to create {directory}: {e}")
    
    # Fix 2: Create basic config if missing
    if not os.path.exists('config.yaml'):
        try:
            basic_config = """
# Basic Elliott Wave Bot Configuration
wave_detection:
  zigzag_threshold: 0.05
  min_wave_length: 5
  confidence_threshold: 0.7

data:
  default_period: "1y"
  default_interval: "1d"

logging:
  level: "INFO"
  file: "logs/elliott_bot.log"
"""
            with open('config.yaml', 'w') as f:
                f.write(basic_config.strip())
            fixes_applied.append("Created basic config.yaml")
        except Exception as e:
            print(f"  ❌ Failed to create config: {e}")
    
    if fixes_applied:
        for fix in fixes_applied:
            print(f"  ✅ {fix}")
    else:
        print("  ✅ No fixes needed")

def generate_health_report():
    """Generate a comprehensive health report"""
    print("\n" + "="*60)
    print("🏥 ELLIOTT WAVE BOT HEALTH REPORT")
    print("="*60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run all checks
    missing_req, missing_opt = check_dependencies()
    data_ok = check_data_loading()
    waves_ok = check_wave_detection()
    viz_ok = check_visualization()
    fib_ok = check_fibonacci_analysis()
    config_ok = check_configuration()
    
    # Apply fixes
    fix_common_issues()
    
    # Overall assessment
    print("\n📊 OVERALL ASSESSMENT:")
    
    core_systems = [data_ok, waves_ok, viz_ok, fib_ok]
    working_systems = sum(core_systems)
    total_systems = len(core_systems)
    
    if working_systems == total_systems and not missing_req:
        status = "🟢 EXCELLENT"
        message = "All systems are working perfectly!"
    elif working_systems >= 3 and not missing_req:
        status = "🟡 GOOD"
        message = "Most systems working, minor issues detected."
    elif working_systems >= 2:
        status = "🟠 NEEDS ATTENTION"
        message = "Some systems have issues, but core functionality works."
    else:
        status = "🔴 CRITICAL"
        message = "Major issues detected, immediate attention required."
    
    print(f"Status: {status}")
    print(f"Core Systems: {working_systems}/{total_systems} working")
    print(f"Assessment: {message}")
    
    if missing_req:
        print(f"\n❌ Missing REQUIRED packages: {', '.join(missing_req)}")
        print("   Install with: pip install " + " ".join(missing_req))
    
    if missing_opt:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_opt)}")
        print("   Install for enhanced features: pip install " + " ".join(missing_opt))
    
    print("\n" + "="*60)
    print("🎯 RECOMMENDATIONS:")
    
    if not data_ok:
        print("  • Check internet connection for Yahoo Finance data")
    if not waves_ok:
        print("  • Run tools/optimize_detection.py to tune wave detection")
    if not viz_ok:
        print("  • Check plotly installation and browser compatibility")
    if not fib_ok:
        print("  • Verify Fibonacci analysis configuration")
    
    if working_systems == total_systems:
        print("  • System is healthy! Consider running performance optimization")
        print("  • Try different timeframes and symbols for analysis")
        print("  • Explore advanced features and customization options")
    
    print("="*60)

if __name__ == "__main__":
    generate_health_report()
