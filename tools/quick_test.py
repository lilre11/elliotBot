"""
Quick Test - Verify Elliott Wave Bot is Working
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def quick_test():
    """Run a quick test of the system"""
    print("🧪 Quick System Test")
    print("=" * 30)
    
    try:
        # Test imports
        print("1️⃣ Testing imports...")
        from src.data.data_loader import DataLoader
        from src.analysis.wave_detector import WaveDetector
        from src.visualization.visualizer import WaveVisualizer
        print("   ✅ All imports successful")
        
        # Test data loading
        print("2️⃣ Testing data loading...")
        loader = DataLoader()
        data = loader.get_yahoo_data('AAPL', period='1mo')
        print(f"   ✅ Loaded {len(data)} data points for AAPL")
        
        # Test wave detection
        print("3️⃣ Testing wave detection...")
        detector = WaveDetector()
        waves = detector.detect_waves(data)
        print(f"   ✅ Detected {len(waves)} waves")
        
        # Test visualization
        print("4️⃣ Testing visualization...")
        visualizer = WaveVisualizer()
        fig = visualizer.plot_waves(data, waves, title="Quick Test")
        fig.write_html("quick_test.html")
        print("   ✅ Created quick_test.html")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("📁 Check 'quick_test.html' for results")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("   • Run the fix_all_problems.py script")
        print("   • Check your internet connection")
        print("   • Try a different symbol or time period")
        return False

if __name__ == "__main__":
    quick_test()
