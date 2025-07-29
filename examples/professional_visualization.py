"""
Enhanced Elliott Wave Visualization Example
Demonstrates comprehensive Elliott Wave visualization with:
- Interactive candlestick charts using plotly.graph_objects
- Wave point annotations (1,2,3,4,5,A,B,C) at swing highs/lows
- Fibonacci retracement lines
- Volume analysis
- Save to HTML file for interactive viewing
"""

import sys
import os
import warnings
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import DataLoader
from src.analysis.wave_detector import WaveDetector
from src.analysis.fibonacci import FibonacciAnalyzer
from src.visualization.visualizer import WaveVisualizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

warnings.filterwarnings('ignore')


def create_professional_chart(data, waves, symbol="STOCK"):
    """
    Create a professional Elliott Wave chart with enhanced features
    """
    print(f"Creating professional visualization for {symbol}...")
    
    # Ensure we have the right column names
    if 'Open' in data.columns:
        # Convert to lowercase if needed
        data = data.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume'
        })
    
    # Create subplot with volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol} - Elliott Wave Analysis', 'Volume'),
        row_heights=[0.8, 0.2]
    )
    
    # 1. Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name=symbol,
            increasing_line_color='#00D4AA',
            decreasing_line_color='#FF6692',
            increasing_fillcolor='rgba(0, 212, 170, 0.3)',
            decreasing_fillcolor='rgba(255, 102, 146, 0.3)',
            line_width=1
        ),
        row=1, col=1
    )
    
    # 2. Add volume bars with color coding
    volume_colors = []
    for i in range(len(data)):
        if data['close'].iloc[i] >= data['open'].iloc[i]:
            volume_colors.append('#00D4AA')  # Green for up days
        else:
            volume_colors.append('#FF6692')  # Red for down days
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['volume'],
            name='Volume',
            marker_color=volume_colors,
            opacity=0.6,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 3. Add Elliott Wave annotations and lines
    if waves:
        print(f"Adding {len(waves)} Elliott Wave annotations...")
        
        # Color scheme for different wave types
        wave_colors = {
            '1': '#FF4757', '2': '#2ED573', '3': '#3742FA', 
            '4': '#FFA502', '5': '#FF3838',
            'A': '#A4B0BE', 'B': '#57606F', 'C': '#2F3542'
        }
        
        # Track wave points for connecting lines
        wave_points = []
        
        for i, wave in enumerate(waves):
            # Extract wave information
            start_time = wave.start_point.timestamp
            end_time = wave.end_point.timestamp
            start_price = wave.start_point.price
            end_price = wave.end_point.price
            wave_type = wave.wave_type.value
            
            # Simplify wave label
            if '_' in wave_type:
                wave_label = wave_type.split('_')[-1]
            else:
                wave_label = wave_type
            
            color = wave_colors.get(wave_label, '#FFD93D')
            
            # Store wave points
            wave_points.extend([(start_time, start_price), (end_time, end_price)])
            
            # Add wave line
            fig.add_trace(
                go.Scatter(
                    x=[start_time, end_time],
                    y=[start_price, end_price],
                    mode='lines+markers',
                    line=dict(color=color, width=3),
                    marker=dict(size=8, color=color, line=dict(width=2, color='white')),
                    name=f'Wave {wave_label}',
                    showlegend=True,
                    hovertemplate=f'<b>Wave {wave_label}</b><br>' +
                                  'Start: %{x}<br>' +
                                  'Price: $%{y:.2f}<br>' +
                                  f'Confidence: {wave.confidence:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add wave label annotation at midpoint
            mid_time = start_time + (end_time - start_time) / 2
            mid_price = (start_price + end_price) / 2
            
            # Adjust annotation position based on wave direction
            if end_price > start_price:
                # Upward wave - place annotation above
                annotation_y = max(start_price, end_price) * 1.02
                arrow_direction = 'down'
            else:
                # Downward wave - place annotation below
                annotation_y = min(start_price, end_price) * 0.98
                arrow_direction = 'up'
            
            fig.add_annotation(
                x=mid_time,
                y=annotation_y,
                text=f"<b>{wave_label}</b>",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=color,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor=color,
                borderwidth=2,
                font=dict(size=16, color=color, family="Arial Black"),
                row=1, col=1
            )
        
        # Connect all wave points with a trend line
        if len(wave_points) > 1:
            wave_x = [point[0] for point in wave_points]
            wave_y = [point[1] for point in wave_points]
            
            fig.add_trace(
                go.Scatter(
                    x=wave_x,
                    y=wave_y,
                    mode='lines',
                    line=dict(color='rgba(123, 123, 123, 0.6)', width=2, dash='dot'),
                    name='Wave Trend',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
    
    # 4. Add key Fibonacci levels (if we have enough data)
    if len(data) > 50:
        period_high = data['high'].max()
        period_low = data['low'].min()
        price_range = period_high - period_low
        
        # Common Fibonacci retracement levels
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        fib_colors = ['#FF9999', '#FFB366', '#FFFF99', '#99FF99', '#99CCFF']
        
        for level, color in zip(fib_levels, fib_colors):
            fib_price = period_high - (price_range * level)
            
            fig.add_hline(
                y=fib_price,
                line_dash="dash",
                line_color=color,
                line_width=1,
                opacity=0.7,
                annotation_text=f"Fib {level*100:.1f}% (${fib_price:.2f})",
                annotation_position="right",
                annotation=dict(font_size=10),
                row=1, col=1
            )
    
    # 5. Enhance layout
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol} - Professional Elliott Wave Analysis</b><br>" +
                 f"<sub>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</sub>",
            x=0.5,
            font=dict(size=24, family="Arial Black")
        ),
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        height=900,
        hovermode='x unified',
        font=dict(family="Arial", size=12)
    )
    
    # Configure axes
    fig.update_xaxes(
        title_text="Date",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)',
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text="Price ($)",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Volume",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)',
        row=2, col=1
    )
    
    # Remove range slider for cleaner appearance
    fig.update_layout(xaxis_rangeslider_visible=False)
    
    return fig


def main():
    """Main function demonstrating enhanced Elliott Wave visualization"""
    
    print("🎨 Enhanced Elliott Wave Visualization Example")
    print("=" * 60)
    
    # Configuration
    symbols = ['AAPL', 'TSLA']  # Two symbols for demonstration
    period = '1y'  # Longer period for better wave detection
    
    # Initialize components
    data_loader = DataLoader()
    wave_detector = WaveDetector()
    fibonacci_analyzer = FibonacciAnalyzer()
    
    for symbol in symbols:
        print(f"\n{'🔍 ANALYZING ' + symbol:^50}")
        print("=" * 50)
        
        try:
            # Load data
            print(f"📊 Loading {symbol} data ({period} period)...")
            data = data_loader.get_yahoo_data(symbol, period=period)
            print(f"✅ Loaded {len(data)} data points")
            
            if len(data) < 50:
                print(f"⚠️  Insufficient data for {symbol}, skipping...")
                continue
            
            # Detect Elliott Waves
            print("🌊 Detecting Elliott Waves...")
            waves = wave_detector.detect_waves(data)
            print(f"✅ Detected {len(waves)} Elliott Waves")
            
            if waves:
                print("\n📋 Wave Summary:")
                for i, wave in enumerate(waves, 1):
                    wave_label = wave.wave_type.value.split('_')[-1] if '_' in wave.wave_type.value else wave.wave_type.value
                    direction = "📈 UP" if wave.direction.value == 1 else "📉 DOWN"
                    print(f"   {i:2d}. Wave {wave_label:2s} | {direction} | "
                          f"Confidence: {wave.confidence:.2f} | "
                          f"{wave.start_point.timestamp.strftime('%m/%d')} → "
                          f"{wave.end_point.timestamp.strftime('%m/%d')}")
            
            # Create enhanced visualization
            print("\n🎨 Creating professional chart...")
            fig = create_professional_chart(data, waves, symbol)
            
            # Save chart
            filename = f"{symbol.lower()}_elliott_waves_professional.html"
            print(f"💾 Saving chart as '{filename}'...")
            fig.write_html(filename)
            print("✅ Chart saved successfully!")
            
            # Display summary statistics
            current_price = data['close'].iloc[-1]
            period_high = data['high'].max()
            period_low = data['low'].min()
            
            print(f"\n📈 Market Summary for {symbol}:")
            print(f"   💰 Current Price: ${current_price:.2f}")
            print(f"   📊 Period Range: ${period_low:.2f} - ${period_high:.2f}")
            print(f"   📏 Price Range: {((period_high - period_low) / period_low * 100):.1f}%")
            print(f"   🎯 Current Position: {((current_price - period_low) / (period_high - period_low) * 100):.1f}% of range")
            
            if waves:
                latest_wave = waves[-1]
                latest_label = latest_wave.wave_type.value.split('_')[-1] if '_' in latest_wave.wave_type.value else latest_wave.wave_type.value
                print(f"   🌊 Latest Wave: {latest_label} ({latest_wave.direction.value})")
                print(f"   🎲 Wave Confidence: {latest_wave.confidence:.2f}")
        
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {str(e)}")
            continue
    
    print(f"\n{'🎉 VISUALIZATION COMPLETE':^60}")
    print("=" * 60)
    print("📁 Generated HTML files can be opened in your browser")
    print("🌐 Each chart is fully interactive with zoom, pan, and hover features")
    print("📊 Features include:")
    print("   • Interactive candlestick charts")
    print("   • Labeled Elliott Wave annotations (1,2,3,4,5,A,B,C)")
    print("   • Fibonacci retracement levels")
    print("   • Volume analysis with color coding")
    print("   • Professional styling and layout")
    print("=" * 60)


if __name__ == "__main__":
    main()
