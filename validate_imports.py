"""
Validation script to test that all imports work correctly
and that Pylance errors are resolved.
"""

def test_ccxt_import():
    """Test ccxt import handling"""
    print("🔍 Testing CCXT import handling...")
    
    try:
        # This should work whether ccxt is installed or not
        from src.data.data_loader import DataLoader, CCXT_AVAILABLE
        
        print(f"✅ DataLoader imported successfully")
        print(f"✅ CCXT_AVAILABLE = {CCXT_AVAILABLE}")
        
        # Test DataLoader instantiation
        loader = DataLoader()
        print(f"✅ DataLoader created successfully")
        print(f"✅ Exchange count: {len(loader.exchanges)}")
        
        if CCXT_AVAILABLE:
            print("✅ CCXT is available - cryptocurrency features enabled")
        else:
            print("ℹ️  CCXT not available - only Yahoo Finance data loading enabled")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic data loading functionality"""
    print("\n📊 Testing basic data loading...")
    
    try:
        from src.data.data_loader import DataLoader
        
        loader = DataLoader()
        
        # Test with a small dataset
        data = loader.get_yahoo_data("AAPL", period="5d")
        print(f"✅ Loaded {len(data)} data points for AAPL")
        print(f"✅ Columns: {list(data.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def main():
    """Main validation function"""
    print("🧪 Elliott Wave Bot - Import Validation")
    print("=" * 50)
    
    test1_passed = test_ccxt_import()
    test2_passed = test_basic_functionality()
    
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Pylance import errors should be resolved")
        print("✅ Elliott Wave Bot is ready to use")
    else:
        print("❌ Some tests failed")
        print("💡 Check the error messages above")

if __name__ == "__main__":
    main()
