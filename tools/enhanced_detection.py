"""
Enhanced Elliott Wave Detection with Multiple Timeframes
This fixes the issue of low wave detection by using adaptive parameters
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import DataLoader
from src.analysis.wave_detector import WaveDetector
from src.visualization.visualizer import WaveVisualizer

def enhanced_wave_analysis(symbol, period='1y'):
    """
    Enhanced wave analysis with multiple sensitivity levels
    """
    print(f"🔍 Enhanced Elliott Wave Analysis for {symbol}")
    print("=" * 50)
    
    loader = DataLoader()
    visualizer = WaveVisualizer()
    
    # Load data
    print(f"📊 Loading {symbol} data ({period} period)...")
    data = loader.get_yahoo_data(symbol, period=period)
    print(f"✅ Loaded {len(data)} data points")
    
    # Try different sensitivity levels
    sensitivity_levels = [
        (0.03, "High Sensitivity"),
        (0.04, "Medium-High Sensitivity"), 
        (0.05, "Standard Sensitivity"),
        (0.06, "Medium-Low Sensitivity"),
        (0.07, "Low Sensitivity")
    ]
    
    best_result = None
    best_score = 0
    
    print(f"\n🌊 Testing Wave Detection Sensitivity Levels:")
    print("-" * 50)
    
    for threshold, description in sensitivity_levels:
        detector = WaveDetector()
        detector.zigzag_threshold = threshold
        detector.min_wave_length = max(3, len(data) // 50)  # Adaptive minimum length
        
        waves = detector.detect_waves(data)
        avg_confidence = sum(w.confidence for w in waves) / len(waves) if waves else 0
        
        # Score based on number of waves and confidence
        score = len(waves) * avg_confidence if waves else 0
        
        print(f"{description:20s} | {len(waves):2d} waves | Avg Confidence: {avg_confidence:.3f}")
        
        if score > best_score and len(waves) > 0:
            best_score = score
            best_result = (threshold, waves, description)
    
    print("-" * 50)
    
    if best_result:
        threshold, waves, description = best_result
        print(f"✅ Best Result: {description} (threshold: {threshold:.2f})")
        print(f"🎯 Detected {len(waves)} waves with average confidence {sum(w.confidence for w in waves) / len(waves):.3f}")
        
        # Show wave details
        print(f"\n📋 Detected Wave Details:")
        for i, wave in enumerate(waves, 1):
            wave_label = wave.wave_type.value.split('_')[-1] if '_' in wave.wave_type.value else wave.wave_type.value
            direction = "📈 UP" if wave.direction.value == 1 else "📉 DOWN"
            start_date = wave.start_point.timestamp.strftime('%m/%d')
            end_date = wave.end_point.timestamp.strftime('%m/%d')
            price_change = ((wave.end_point.price - wave.start_point.price) / wave.start_point.price) * 100
            
            print(f"   {i:2d}. Wave {wave_label:2s} | {direction} | {start_date} → {end_date} | "
                  f"{price_change:+5.1f}% | Confidence: {wave.confidence:.2f}")
        
        # Create visualization
        print(f"\n🎨 Creating enhanced visualization...")
        fig = visualizer.plot_waves(data, waves, title=f"{symbol} - Enhanced Elliott Wave Analysis")
        
        # Save chart
        filename = f"{symbol.lower()}_enhanced_waves.html"
        fig.write_html(filename)
        print(f"✅ Chart saved as '{filename}'")
        
        return waves, data, fig
    else:
        print("❌ No waves detected with any sensitivity level")
        print("💡 Suggestions:")
        print("   • Try a longer time period (6mo, 1y, 2y)")
        print("   • Use a more volatile symbol")
        print("   • Check if the symbol has sufficient price movement")
        
        return [], data, None

def main():
    """Main function for enhanced wave analysis"""
    
    print("🚀 Enhanced Elliott Wave Detection System")
    print("=" * 50)
    print("This system automatically finds the best sensitivity")
    print("settings for optimal wave detection.\n")
    
    # Test with multiple symbols
    symbols = ['AAPL', 'TSLA', 'NVDA', 'BTC-USD']
    period = '1y'
    
    results = {}
    
    for symbol in symbols:
        try:
            print(f"\n{'='*60}")
            waves, data, fig = enhanced_wave_analysis(symbol, period)
            results[symbol] = {
                'waves': len(waves),
                'data_points': len(data),
                'success': len(waves) > 0
            }
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {str(e)}")
            results[symbol] = {'waves': 0, 'data_points': 0, 'success': False}
    
    # Summary report
    print(f"\n{'='*60}")
    print("📊 ENHANCED ANALYSIS SUMMARY")
    print("=" * 60)
    
    for symbol, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ NO WAVES"
        print(f"{symbol:8s} | {result['data_points']:3d} data points | "
              f"{result['waves']:2d} waves | {status}")
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"\nSuccess Rate: {successful}/{total} symbols ({successful/total*100:.1f}%)")
    
    if successful > 0:
        print(f"\n🎉 Enhanced wave detection is working!")
        print(f"📁 Check the generated HTML files for interactive charts")
    else:
        print(f"\n⚠️  No waves detected for any symbol")
        print(f"💡 This might indicate:")
        print(f"   • Market is in a consolidation phase")
        print(f"   • Need longer time periods")
        print(f"   • Different symbols might work better")

if __name__ == "__main__":
    main()
