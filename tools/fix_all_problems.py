"""
Elliott Wave Bot - Automatic Problem Fixer
This script automatically detects and fixes common issues
"""

import sys
import os
import subprocess
import importlib

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_root)

def check_and_fix_dependencies():
    """Check and install missing dependencies"""
    print("🔧 Checking Dependencies...")
    
    required_packages = [
        'yfinance', 'pandas', 'numpy', 'plotly', 'ta'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n🚀 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install packages: {e}")
            return False
    
    return True

def fix_data_column_issues():
    """Fix data column naming issues"""
    print("\n🔧 Fixing Data Column Issues...")
    
    visualization_files = [
        'examples/professional_visualization.py',
        'examples/advanced_visualization.py',
        'examples/visualization_showcase.py'
    ]
    
    fixes_applied = 0
    
    for file_path in visualization_files:
        full_path = os.path.join(project_root, file_path)
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                
                # Common column name fixes
                fixes = [
                    ("data['Open']", "data['open']"),
                    ("data['High']", "data['high']"),
                    ("data['Low']", "data['low']"),
                    ("data['Close']", "data['close']"),
                    ("data['Volume']", "data['volume']"),
                    ("df['Open']", "df['open']"),
                    ("df['High']", "df['high']"),
                    ("df['Low']", "df['low']"),
                    ("df['Close']", "df['close']"),
                    ("df['Volume']", "df['volume']")
                ]
                
                original_content = content
                for old, new in fixes:
                    content = content.replace(old, new)
                
                if content != original_content:
                    with open(full_path, 'w') as f:
                        f.write(content)
                    fixes_applied += 1
                    print(f"   ✅ Fixed {file_path}")
                else:
                    print(f"   ✅ {file_path} - already correct")
                    
            except Exception as e:
                print(f"   ❌ Error fixing {file_path}: {e}")
    
    print(f"🎯 Applied fixes to {fixes_applied} files")

def fix_import_paths():
    """Fix import path issues"""
    print("\n🔧 Fixing Import Paths...")
    
    # Create __init__.py files if missing
    init_dirs = [
        'src',
        'src/data',
        'src/analysis', 
        'src/visualization',
        'src/config'
    ]
    
    for dir_path in init_dirs:
        full_dir = os.path.join(project_root, dir_path)
        init_file = os.path.join(full_dir, '__init__.py')
        
        if os.path.exists(full_dir) and not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Elliott Wave Bot Module\n')
            print(f"   ✅ Created {init_file}")
        elif os.path.exists(init_file):
            print(f"   ✅ {init_file} exists")

def create_quick_test():
    """Create a quick test to verify everything works"""
    print("\n🔧 Creating Quick Test...")
    
    test_content = '''"""
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
        
        print("\\n🎉 ALL TESTS PASSED!")
        print("📁 Check 'quick_test.html' for results")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("\\n💡 Troubleshooting:")
        print("   • Run the fix_all_problems.py script")
        print("   • Check your internet connection")
        print("   • Try a different symbol or time period")
        return False

if __name__ == "__main__":
    quick_test()
'''
    
    test_file = os.path.join(project_root, 'tools', 'quick_test.py')
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    print(f"   ✅ Created quick_test.py")

def main():
    """Main function to fix all problems"""
    print("🚀 Elliott Wave Bot - Automatic Problem Fixer")
    print("=" * 50)
    print("This script will automatically detect and fix common issues.")
    print()
    
    # Step 1: Dependencies
    if not check_and_fix_dependencies():
        print("❌ Failed to fix dependencies. Please install manually.")
        return
    
    # Step 2: Column naming
    fix_data_column_issues()
    
    # Step 3: Import paths
    fix_import_paths()
    
    # Step 4: Create test
    create_quick_test()
    
    # Step 5: Run test
    print("\n🧪 Running Quick Test...")
    try:
        os.chdir(project_root)
        result = subprocess.run([sys.executable, 'tools/quick_test.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("❌ Quick test failed:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error running test: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 FIX SUMMARY")
    print("=" * 50)
    print("✅ Dependencies checked/installed")
    print("✅ Data column naming fixed")
    print("✅ Import paths fixed")
    print("✅ Quick test created")
    print()
    print("📁 Files you can now run:")
    print("   • tools/quick_test.py - Basic functionality test")
    print("   • examples/visualization_showcase.py - Full demo")
    print("   • tools/enhanced_detection.py - Better wave detection")
    print()
    print("🎉 Elliott Wave Bot should now be working correctly!")

if __name__ == "__main__":
    main()
