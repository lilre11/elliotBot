"""
Visualization module for Elliott Wave analysis using Plotly.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Dict, Any, Optional, Tuple
import logging

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..analysis.wave_detector import Wave, WaveType, TrendDirection
from ..analysis.fibonacci import FibonacciAnalysis, FibonacciLevel

logger = get_logger(__name__)


class WaveVisualizer:
    """
    Main visualization class for Elliott Wave analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize visualizer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.theme = self.config.get('visualization.default_theme', 'plotly_dark')
        self.show_fibonacci = self.config.get('visualization.show_fibonacci', True)
        self.show_volume = self.config.get('visualization.show_volume', True)
        self.chart_height = self.config.get('visualization.chart_height', 800)
        self.chart_width = self.config.get('visualization.chart_width', 1200)
        
        # Color scheme
        self.colors = {
            'impulse_up': '#00ff00',      # Green for upward impulse waves
            'impulse_down': '#ff0000',    # Red for downward impulse waves
            'corrective_up': '#90EE90',   # Light green for upward corrective waves
            'corrective_down': '#FFA500', # Orange for downward corrective waves
            'fibonacci': '#FFD700',       # Gold for Fibonacci levels
            'support': '#00BFFF',         # Deep sky blue for support
            'resistance': '#FF69B4',      # Hot pink for resistance
            'volume': '#808080'           # Gray for volume
        }
        
        logger.info("WaveVisualizer initialized")
    
    def plot_waves(
        self, 
        data: pd.DataFrame, 
        waves: List[Wave],
        fibonacci_analysis: Optional[FibonacciAnalysis] = None,
        title: str = "Elliott Wave Analysis",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create main Elliott Wave chart with price data and wave annotations.
        
        Args:
            data: OHLCV DataFrame
            waves: List of detected waves
            fibonacci_analysis: Optional Fibonacci analysis
            title: Chart title
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure
        """
        try:
            # Create subplots
            if self.show_volume:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(title, "Volume")
                )
            else:
                fig = go.Figure()
            
            # Add candlestick chart
            candlestick = go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price",
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff0000'
            )
            
            if self.show_volume:
                fig.add_trace(candlestick, row=1, col=1)
            else:
                fig.add_trace(candlestick)
            
            # Add wave annotations
            if waves:
                self._add_wave_annotations(fig, waves, data, row=1 if self.show_volume else None)
            
            # Add Fibonacci levels
            if fibonacci_analysis and self.show_fibonacci:
                self._add_fibonacci_levels(fig, fibonacci_analysis, data, row=1 if self.show_volume else None)
            
            # Add volume if enabled
            if self.show_volume:
                self._add_volume_chart(fig, data, row=2)
            
            # Update layout
            self._update_layout(fig, title)
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Chart saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating wave chart: {e}")
            raise
    
    def _add_wave_annotations(self, fig: go.Figure, waves: List[Wave], data: pd.DataFrame, row: Optional[int] = None):
        """
        Add wave lines and labels to the chart.
        
        Args:
            fig: Plotly figure
            waves: List of waves
            data: OHLCV DataFrame
            row: Row number for subplot
        """
        for wave in waves:
            # Determine color based on wave type and direction
            color = self._get_wave_color(wave)
            
            # Draw wave line
            fig.add_trace(
                go.Scatter(
                    x=[wave.start_point.timestamp, wave.end_point.timestamp],
                    y=[wave.start_point.price, wave.end_point.price],
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=f"Wave {wave.wave_type.value}",
                    showlegend=False,
                    hovertemplate=f"Wave {wave.wave_type.value}<br>" +
                                 f"Confidence: {wave.confidence:.2f}<br>" +
                                 f"Start: {wave.start_point.price:.2f}<br>" +
                                 f"End: {wave.end_point.price:.2f}<br>" +
                                 f"Change: {wave.price_change_pct:.2%}<extra></extra>"
                ),
                row=row, col=1
            )
            
            # Add wave label
            mid_timestamp = wave.start_point.timestamp + (wave.end_point.timestamp - wave.start_point.timestamp) / 2
            mid_price = (wave.start_point.price + wave.end_point.price) / 2
            
            fig.add_annotation(
                x=mid_timestamp,
                y=mid_price,
                text=f"<b>{wave.wave_type.value}</b>",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=color,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor=color,
                font=dict(size=12, color='black'),
                row=row,
                col=1
            )
    
    def _get_wave_color(self, wave: Wave) -> str:
        """
        Get color for wave based on type and direction.
        
        Args:
            wave: Wave object
            
        Returns:
            Color string
        """
        if wave.wave_type in [WaveType.IMPULSE_1, WaveType.IMPULSE_3, WaveType.IMPULSE_5]:
            return self.colors['impulse_up'] if wave.direction == TrendDirection.UP else self.colors['impulse_down']
        else:
            return self.colors['corrective_up'] if wave.direction == TrendDirection.UP else self.colors['corrective_down']
    
    def _add_fibonacci_levels(self, fig: go.Figure, fib_analysis: FibonacciAnalysis, data: pd.DataFrame, row: Optional[int] = None):
        """
        Add Fibonacci retracement and extension levels to the chart.
        
        Args:
            fig: Plotly figure
            fib_analysis: Fibonacci analysis
            data: OHLCV DataFrame
            row: Row number for subplot
        """
        x_range = [data.index[0], data.index[-1]]
        
        # Add key Fibonacci levels
        for level in fib_analysis.key_levels:
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[level.price, level.price],
                    mode='lines',
                    line=dict(
                        color=self.colors['fibonacci'],
                        width=1,
                        dash='dash' if level.level_type == 'extension' else 'solid'
                    ),
                    name=f"Fib {level.ratio:.3f}",
                    showlegend=False,
                    hovertemplate=f"Fibonacci {level.ratio:.1%}<br>" +
                                 f"Price: {level.price:.2f}<br>" +
                                 f"Type: {level.level_type}<extra></extra>"
                ),
                row=row, col=1
            )
            
            # Add level label
            fig.add_annotation(
                x=data.index[-1],
                y=level.price,
                text=f"{level.ratio:.1%}",
                showarrow=False,
                xanchor='left',
                bgcolor='rgba(255,215,0,0.8)',
                font=dict(size=10, color='black'),
                row=row,
                col=1
            )
    
    def _add_volume_chart(self, fig: go.Figure, data: pd.DataFrame, row: int):
        """
        Add volume chart to subplot.
        
        Args:
            fig: Plotly figure
            data: OHLCV DataFrame
            row: Row number
        """
        # Color volume bars based on price movement
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(data['close'], data['open'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name="Volume",
                marker_color=colors,
                opacity=0.7,
                showlegend=False
            ),
            row=row, col=1
        )
    
    def _update_layout(self, fig: go.Figure, title: str):
        """
        Update chart layout with theme and styling.
        
        Args:
            fig: Plotly figure
            title: Chart title
        """
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            template=self.theme,
            height=self.chart_height,
            width=self.chart_width,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Price", row=1, col=1)
        
        if self.show_volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    def plot_fibonacci_analysis(
        self, 
        data: pd.DataFrame, 
        fib_analysis: FibonacciAnalysis,
        title: str = "Fibonacci Analysis"
    ) -> go.Figure:
        """
        Create dedicated Fibonacci analysis chart.
        
        Args:
            data: OHLCV DataFrame
            fib_analysis: Fibonacci analysis
            title: Chart title
            
        Returns:
            Plotly figure
        """
        try:
            fig = go.Figure()
            
            # Add price data
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=2)
                )
            )
            
            # Add swing points
            fig.add_trace(
                go.Scatter(
                    x=[data.index[0], data.index[-1]],  # Approximate positions
                    y=[fib_analysis.swing_low, fib_analysis.swing_high],
                    mode='markers',
                    name='Swing Points',
                    marker=dict(size=10, color='red', symbol='diamond')
                )
            )
            
            # Add all Fibonacci levels
            x_range = [data.index[0], data.index[-1]]
            
            for level in fib_analysis.retracements + fib_analysis.extensions:
                line_style = 'solid' if level.is_key_level else 'dot'
                line_width = 2 if level.is_key_level else 1
                
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=[level.price, level.price],
                        mode='lines',
                        line=dict(
                            color=self.colors['fibonacci'],
                            width=line_width,
                            dash=line_style
                        ),
                        name=f"Fib {level.ratio:.1%}",
                        showlegend=level.is_key_level
                    )
                )
            
            # Highlight current price
            fig.add_hline(
                y=fib_analysis.current_price,
                line_dash="dash",
                line_color="white",
                annotation_text=f"Current: {fib_analysis.current_price:.2f}"
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                template=self.theme,
                height=600,
                width=self.chart_width,
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Fibonacci chart: {e}")
            raise
    
    def plot_wave_progression(self, waves: List[Wave], title: str = "Wave Progression") -> go.Figure:
        """
        Create a chart showing wave progression over time.
        
        Args:
            waves: List of waves
            title: Chart title
            
        Returns:
            Plotly figure
        """
        try:
            fig = go.Figure()
            
            if not waves:
                fig.add_annotation(
                    text="No waves detected",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
                return fig
            
            # Create wave progression line
            x_points = []
            y_points = []
            wave_labels = []
            
            for i, wave in enumerate(waves):
                if i == 0:
                    x_points.append(wave.start_point.timestamp)
                    y_points.append(wave.start_point.price)
                    wave_labels.append(f"Start")
                
                x_points.append(wave.end_point.timestamp)
                y_points.append(wave.end_point.price)
                wave_labels.append(f"Wave {wave.wave_type.value}")
            
            # Add main progression line
            fig.add_trace(
                go.Scatter(
                    x=x_points,
                    y=y_points,
                    mode='lines+markers',
                    name='Wave Progression',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8, color='red')
                )
            )
            
            # Add wave labels
            for i, (x, y, label) in enumerate(zip(x_points[1:], y_points[1:], wave_labels[1:])):
                fig.add_annotation(
                    x=x, y=y,
                    text=label,
                    showarrow=True,
                    arrowhead=2,
                    bgcolor='rgba(255,255,255,0.8)',
                    font=dict(size=10)
                )
            
            # Update layout
            fig.update_layout(
                title=title,
                template=self.theme,
                height=500,
                width=self.chart_width,
                xaxis_title="Date",
                yaxis_title="Price",
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating wave progression chart: {e}")
            raise
    
    def create_dashboard(
        self, 
        data: pd.DataFrame, 
        waves: List[Wave],
        fibonacci_analysis: Optional[FibonacciAnalysis] = None,
        additional_indicators: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """
        Create comprehensive dashboard with multiple charts.
        
        Args:
            data: OHLCV DataFrame
            waves: List of detected waves
            fibonacci_analysis: Optional Fibonacci analysis
            additional_indicators: Optional technical indicators
            
        Returns:
            Plotly figure with dashboard layout
        """
        try:
            # Create subplot structure
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    "Elliott Wave Analysis", "Wave Progression",
                    "Fibonacci Levels", "RSI & MACD",
                    "Volume Analysis", "Wave Confidence"
                ),
                specs=[
                    [{"colspan": 2}, None],
                    [{"secondary_y": True}, {"secondary_y": True}],
                    [{"secondary_y": False}, {"secondary_y": False}]
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.05
            )
            
            # 1. Main Elliott Wave chart (top row, full width)
            candlestick = go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price"
            )
            fig.add_trace(candlestick, row=1, col=1)
            
            if waves:
                self._add_wave_annotations(fig, waves, data, row=1)
            
            if fibonacci_analysis:
                self._add_fibonacci_levels(fig, fibonacci_analysis, data, row=1)
            
            # 2. Wave progression (row 2, col 1)
            if waves:
                wave_times = [w.end_point.timestamp for w in waves]
                wave_prices = [w.end_point.price for w in waves]
                wave_types = [w.wave_type.value for w in waves]
                
                fig.add_trace(
                    go.Scatter(
                        x=wave_times,
                        y=wave_prices,
                        mode='lines+markers',
                        name='Wave Progression',
                        text=wave_types,
                        textposition='top center'
                    ),
                    row=2, col=1
                )
            
            # 3. Technical indicators (row 2, col 2)
            if additional_indicators is not None:
                if 'rsi' in additional_indicators.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=additional_indicators.index,
                            y=additional_indicators['rsi'],
                            name='RSI',
                            line=dict(color='orange')
                        ),
                        row=2, col=2
                    )
                    
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=2)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=2)
            
            # 4. Volume analysis (row 3, col 1)
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['volume'],
                    name='Volume',
                    opacity=0.6
                ),
                row=3, col=1
            )
            
            # 5. Wave confidence (row 3, col 2)
            if waves:
                confidences = [w.confidence for w in waves]
                wave_labels = [w.wave_type.value for w in waves]
                
                fig.add_trace(
                    go.Bar(
                        x=wave_labels,
                        y=confidences,
                        name='Wave Confidence',
                        marker_color='lightblue'
                    ),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                title="Elliott Wave Trading Dashboard",
                template=self.theme,
                height=1000,
                width=self.chart_width,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            raise
    
    def save_chart(self, fig: go.Figure, filepath: str, format: str = 'html'):
        """
        Save chart to file.
        
        Args:
            fig: Plotly figure
            filepath: Output file path
            format: Output format ('html', 'png', 'pdf', 'svg')
        """
        try:
            if format.lower() == 'html':
                fig.write_html(filepath)
            elif format.lower() == 'png':
                fig.write_image(filepath)
            elif format.lower() == 'pdf':
                fig.write_image(filepath)
            elif format.lower() == 'svg':
                fig.write_image(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Chart saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving chart: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.data.data_loader import DataLoader
    from src.analysis.wave_detector import WaveDetector
    from src.analysis.fibonacci import FibonacciAnalyzer
    
    # Load sample data
    loader = DataLoader()
    data = loader.get_yahoo_data("AAPL", period="6mo")
    
    # Detect waves
    detector = WaveDetector()
    waves = detector.detect_waves(data)
    
    # Fibonacci analysis
    fib_analyzer = FibonacciAnalyzer()
    if len(data) > 50:
        high_price = data['high'].rolling(50).max().iloc[-1]
        low_price = data['low'].rolling(50).min().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        fib_analysis = fib_analyzer.analyze_retracement(high_price, low_price, current_price, 'up')
    else:
        fib_analysis = None
    
    # Create visualizations
    visualizer = WaveVisualizer()
    
    # Main wave chart
    fig = visualizer.plot_waves(data, waves, fib_analysis)
    fig.show()
    
    # Wave progression chart
    if waves:
        progression_fig = visualizer.plot_wave_progression(waves)
        progression_fig.show()
    
    # Dashboard
    dashboard = visualizer.create_dashboard(data, waves, fib_analysis)
    dashboard.show()
    
    print("Visualizations created successfully!")
