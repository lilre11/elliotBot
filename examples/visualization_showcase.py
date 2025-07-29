"""
🎨 ULTIMATE ELLIOTT WAVE VISUALIZATION SHOWCASE 🎨

This script demonstrates ALL the advanced visualization features requested:
✅ Interactive candlestick charts using plotly.graph_objects
✅ Labeled wave points (annotations at swing highs/lows: 1, 2, 3, 4, 5, A, B, C)
✅ Fibonacci retracement lines 
✅ Professional styling and interactive features
✅ Save charts to HTML files for browser viewing

Run this script to see the complete Elliott Wave visualization system in action!
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import DataLoader
from src.analysis.wave_detector import WaveDetector
from src.visualization.visualizer import WaveVisualizer

def main():
    print("🚀" + "="*58 + "🚀")
    print("🎨  ELLIOTT WAVE VISUALIZATION SHOWCASE  🎨")
    print("🚀" + "="*58 + "🚀")
    print()
    print("This demonstration shows:")
    print("✅ Interactive candlestick charts with plotly.graph_objects")
    print("✅ Labeled Elliott Wave annotations (1,2,3,4,5,A,B,C)")
    print("✅ Fibonacci retracement levels as horizontal lines")
    print("✅ Professional styling and hover information")
    print("✅ HTML export for browser viewing")
    print()
    
    # Initialize components
    data_loader = DataLoader()
    wave_detector = WaveDetector()
    visualizer = WaveVisualizer()
    
    # Test with AAPL - good for demonstrations
    symbol = "AAPL"
    period = "1y"
    
    print(f"📊 Loading {symbol} data ({period} period)...")
    data = data_loader.get_yahoo_data(symbol, period=period)
    print(f"✅ Loaded {len(data)} data points")
    
    print("🌊 Detecting Elliott Waves...")
    waves = wave_detector.detect_waves(data)
    print(f"✅ Detected {len(waves)} Elliott Waves")
    
    if waves:
        print("\n📋 Detected Wave Summary:")
        for i, wave in enumerate(waves, 1):
            wave_label = wave.wave_type.value.split('_')[-1] if '_' in wave.wave_type.value else wave.wave_type.value
            direction = "📈 UP" if wave.direction.value == 1 else "📉 DOWN"
            start_date = wave.start_point.timestamp.strftime('%m/%d/%y')
            end_date = wave.end_point.timestamp.strftime('%m/%d/%y')
            print(f"   {i:2d}. Wave {wave_label:2s} | {direction} | "
                  f"Confidence: {wave.confidence:.2f} | "
                  f"{start_date} → {end_date}")
    
    print(f"\n🎨 Creating comprehensive Elliott Wave visualization...")
    
    # Create the visualization using the existing WaveVisualizer
    # This includes candlestick charts, wave annotations, and Fibonacci levels
    fig = visualizer.plot_waves(
        data, 
        waves, 
        title=f"{symbol} - Complete Elliott Wave Analysis"
    )
    
    # Save as HTML file
    filename = f"{symbol.lower()}_complete_elliott_wave_showcase.html"
    print(f"💾 Saving comprehensive chart as '{filename}'...")
    fig.write_html(filename)
    print("✅ Chart saved successfully!")
    
    # Display market summary
    current_price = data['close'].iloc[-1]
    period_high = data['high'].max()
    period_low = data['low'].min()
    
    print(f"\n📈 Market Summary for {symbol}:")
    print(f"   💰 Current Price: ${current_price:.2f}")
    print(f"   📊 Period Range: ${period_low:.2f} - ${period_high:.2f}")
    print(f"   📏 Price Swing: {((period_high - period_low) / period_low * 100):.1f}%")
    print(f"   🎯 Current Position: {((current_price - period_low) / (period_high - period_low) * 100):.1f}% of range")
    
    if waves:
        latest_wave = waves[-1]
        latest_label = latest_wave.wave_type.value.split('_')[-1] if '_' in latest_wave.wave_type.value else latest_wave.wave_type.value
        direction_emoji = "📈" if latest_wave.direction.value == 1 else "📉"
        print(f"   🌊 Latest Wave: {latest_label} {direction_emoji}")
        print(f"   🎲 Confidence: {latest_wave.confidence:.2f}")
    
    print(f"\n🎉" + "="*56 + "🎉")
    print("        VISUALIZATION SHOWCASE COMPLETE!")
    print("🎉" + "="*56 + "🎉")
    print()
    print("🌐 Open the generated HTML file in your browser to see:")
    print(f"   📁 File: {filename}")
    print("   🖱️  Interactive candlestick chart")
    print("   🔍 Zoom, pan, and hover features")
    print("   🌊 Elliott Wave annotations (1,2,3,4,5,A,B,C)")
    print("   📐 Fibonacci retracement levels")
    print("   📊 Volume analysis")
    print("   ✨ Professional styling")
    print()
    print("🎨 Your Elliott Wave Bot visualization system is ready! 🎨")
    print("="*64)

if __name__ == "__main__":
    main()
