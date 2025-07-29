"""
Advanced Elliott Wave Visualization Example
This example demonstrates comprehensive Elliott Wave visualization with:
- Interactive candlestick charts
- Wave point annotations (1,2,3,4,5,A,B,C)
- Fibonacci retracement and extension lines
- Multiple display options (HTML save, interactive show)
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


def create_enhanced_chart(data, waves, fibonacci_levels=None, symbol="STOCK"):
    """
    Create an enhanced Elliott Wave chart with all visualization features
    """
    print(f"Creating enhanced visualization for {symbol}...")
    
    # Create subplot with secondary y-axis for volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{symbol} - Elliott Wave Analysis', 'Volume'),
        row_width=[0.7, 0.3]
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
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444',
            increasing_fillcolor='rgba(0, 255, 136, 0.3)',
            decreasing_fillcolor='rgba(255, 68, 68, 0.3)'
        ),
        row=1, col=1
    )
    
    # 2. Add volume bars
    colors = ['red' if close < open else 'green' 
              for close, open in zip(data['close'], data['open'])]
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.6
        ),
        row=2, col=1
    )
    
    # 3. Add wave annotations and lines
    if waves:
        print(f"Adding {len(waves)} wave annotations...")
        
        # Wave colors for different types
        wave_colors = {
            '1': '#FF6B6B', '2': '#4ECDC4', '3': '#45B7D1', 
            '4': '#96CEB4', '5': '#FFEAA7',
            'A': '#DDA0DD', 'B': '#98D8C8', 'C': '#F7DC6F'
        }
        
        # Add wave lines and annotations
        for i, wave in enumerate(waves):
            start_timestamp = wave.start_point.timestamp
            end_timestamp = wave.end_point.timestamp
            start_price = wave.start_point.price
            end_price = wave.end_point.price
            wave_label = wave.wave_type.value
            
            # Clean up the wave label for display
            display_label = wave_label.split('_')[-1] if '_' in wave_label else wave_label
            
            color = wave_colors.get(display_label, '#FFD93D')
            
            # Add wave line
            fig.add_trace(
                go.Scatter(
                    x=[start_timestamp, end_timestamp],
                    y=[start_price, end_price],
                    mode='lines',
                    line=dict(color=color, width=3),
                    name=f'Wave {display_label}',
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Add wave label annotation
            mid_x = start_timestamp + (end_timestamp - start_timestamp) / 2
            mid_y = (start_price + end_price) / 2
            
            fig.add_annotation(
                x=mid_x,
                y=mid_y,
                text=f"<b>{display_label}</b>",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=color,
                bgcolor="white",
                bordercolor=color,
                borderwidth=2,
                font=dict(size=14, color=color),
                row=1, col=1
            )
            
            # Add wave start/end point markers
            fig.add_trace(
                go.Scatter(
                    x=[start_timestamp, end_timestamp],
                    y=[start_price, end_price],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=color,
                        symbol='circle',
                        line=dict(width=2, color='white')
                    ),
                    name=f'Wave {display_label} Points',
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # 3. Add wave annotations and lines
    if waves:
        print(f"Adding {len(waves)} wave annotations...")
        
        # Wave colors for different types
        wave_colors = {
            '1': '#FF6B6B', '2': '#4ECDC4', '3': '#45B7D1', 
            '4': '#96CEB4', '5': '#FFEAA7',
            'A': '#DDA0DD', 'B': '#98D8C8', 'C': '#F7DC6F'
        }
        
        # Add wave lines and annotations
        for i, wave in enumerate(waves):
            start_timestamp = wave.start_point.timestamp
            end_timestamp = wave.end_point.timestamp
            start_price = wave.start_point.price
            end_price = wave.end_point.price
            wave_label = wave.wave_type.value
            
            # Clean up the wave label for display
            display_label = wave_label.split('_')[-1] if '_' in wave_label else wave_label
            
            color = wave_colors.get(display_label, '#FFD93D')
            
            # Add wave line
            fig.add_trace(
                go.Scatter(
                    x=[start_timestamp, end_timestamp],
                    y=[start_price, end_price],
                    mode='lines',
                    line=dict(color=color, width=3),
                    name=f'Wave {display_label}',
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Add wave label annotation
            mid_x = start_timestamp + (end_timestamp - start_timestamp) / 2
            mid_y = (start_price + end_price) / 2
            
            fig.add_annotation(
                x=mid_x,
                y=mid_y,
                text=f"<b>{display_label}</b>",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=color,
                bgcolor="white",
                bordercolor=color,
                borderwidth=2,
                font=dict(size=14, color=color),
                row=1, col=1
            )
            
            # Add wave start/end point markers
            fig.add_trace(
                go.Scatter(
                    x=[start_timestamp, end_timestamp],
                    y=[start_price, end_price],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=color,
                        symbol='circle',
                        line=dict(width=2, color='white')
                    ),
                    name=f'Wave {display_label} Points',
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # 4. Add Fibonacci levels if provided
    if fibonacci_levels:
        print("Adding Fibonacci retracement levels...")
        
        fib_colors = {
            23.6: '#FF9999', 38.2: '#FFB366', 50.0: '#FFFF99',
            61.8: '#99FF99', 78.6: '#99CCFF', 100.0: '#CC99FF'
        }
        
        for level_name, fib_analysis in fibonacci_levels.items():
            if hasattr(fib_analysis, 'retracement_levels'):
                for fib_level in fib_analysis.retracement_levels:
                    color = fib_colors.get(fib_level.ratio * 100, '#DDDDDD')
                    
                    # Add horizontal Fibonacci line
                    fig.add_hline(
                        y=fib_level.price,
                        line_dash="dash",
                        line_color=color,
                        line_width=1,
                        annotation_text=f"Fib {fib_level.ratio*100:.1f}% - ${fib_level.price:.2f}",
                        annotation_position="right",
                        row=1, col=1
                    )
    
    # 5. Customize layout
    fig.update_layout(
        title=dict(
            text=f"{symbol} - Elliott Wave Analysis with Enhanced Visualization",
            x=0.5,
            font=dict(size=20)
        ),
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=800,
        hovermode='x unified'
    )
    
    # Remove rangeslider for cleaner look
    fig.update_layout(xaxis_rangeslider_visible=False)
    
    # Update y-axes
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig


def main():
    """Main function demonstrating advanced Elliott Wave visualization"""
    
    print("=== Advanced Elliott Wave Visualization Example ===")
    
    # Configuration
    symbols = ['AAPL', 'TSLA', 'NVDA']  # Multiple symbols for demonstration
    period = '1y'  # Longer period for better wave detection
    
    # Initialize components
    data_loader = DataLoader()
    wave_detector = WaveDetector()
    fibonacci_analyzer = FibonacciAnalyzer()
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"ANALYZING {symbol}")
        print(f"{'='*50}")
        
        try:
            # Load data
            print(f"Loading {symbol} data for {period} period...")
            data = data_loader.get_yahoo_data(symbol, period=period)
            print(f"Loaded {len(data)} data points")
            
            if len(data) < 50:
                print(f"Insufficient data for {symbol}, skipping...")
                continue
            
            # Detect Elliott Waves
            print("Detecting Elliott Waves...")
            waves = wave_detector.detect_waves(data)
            print(f"Detected {len(waves)} waves")
            
            if not waves:
                print(f"No waves detected for {symbol}, creating basic chart...")
                # Create basic chart without waves
                fig = create_enhanced_chart(data, [], symbol=symbol)
            else:
                # Display detected waves
                print("\nDetected Waves:")
                for wave in waves:
                    wave_label = wave.wave_type.value.split('_')[-1] if '_' in wave.wave_type.value else wave.wave_type.value
                    print(f"  Wave {wave_label}: {wave.direction.value} "
                          f"({wave.start_point.timestamp.strftime('%Y-%m-%d')} -> "
                          f"{wave.end_point.timestamp.strftime('%Y-%m-%d')}) "
                          f"Confidence: {wave.confidence:.2f}")
                
                # Analyze Fibonacci levels
                print("\nCalculating Fibonacci levels...")
                fibonacci_levels = {}
                
                # Calculate Fibonacci for the most recent significant wave
                if waves:
                    recent_wave = waves[-1]
                    wave_high = recent_wave.end_point.price if recent_wave.direction.value == 1 else recent_wave.start_point.price
                    wave_low = recent_wave.start_point.price if recent_wave.direction.value == 1 else recent_wave.end_point.price
                    
                    if wave_high != wave_low:
                        fib_analysis = fibonacci_analyzer.analyze_retracement(
                            wave_high, wave_low, 
                            recent_wave.end_point.price,
                            'UP' if recent_wave.direction.value == 1 else 'DOWN'
                        )
                        fibonacci_levels['recent_wave'] = fib_analysis
                
                # Create enhanced chart
                fig = create_enhanced_chart(data, waves, fibonacci_levels, symbol)
            
            # Save chart
            filename = f"{symbol.lower()}_elliott_wave_enhanced.html"
            print(f"\nSaving enhanced chart as '{filename}'...")
            fig.write_html(filename)
            print(f"✅ Chart saved successfully!")
            
            # Option to show interactively (commented out to avoid nbformat issues)
            # Uncomment the line below if you have nbformat installed and want to show charts
            # fig.show()
            
            # Display summary statistics
            print(f"\n📊 Summary for {symbol}:")
            print(f"  📈 Current Price: ${data['close'].iloc[-1]:.2f}")
            print(f"  📉 Period Low: ${data['low'].min():.2f}")
            print(f"  📈 Period High: ${data['high'].max():.2f}")
            print(f"  🌊 Waves Detected: {len(waves)}")
            
            if waves:
                latest_wave = waves[-1]
                latest_label = latest_wave.wave_type.value.split('_')[-1] if '_' in latest_wave.wave_type.value else latest_wave.wave_type.value
                print(f"  🎯 Latest Wave: {latest_label} ({latest_wave.direction.value})")
                print(f"  🎲 Confidence: {latest_wave.confidence:.2f}")
        
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("🎉 Advanced visualization example completed!")
    print("📁 Check the generated HTML files for interactive charts")
    print("🌐 Open them in your browser to explore the visualizations")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
