"""
Example: Real-time Analysis
This example demonstrates real-time Elliott Wave analysis and monitoring.
"""

import sys
import os
import time
import warnings
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.data_loader import DataLoader
from src.analysis.wave_detector import WaveDetector
from src.analysis.fibonacci import FibonacciAnalyzer
from src.trading.strategy import ElliottWaveStrategy
from src.visualization.visualizer import WaveVisualizer

warnings.filterwarnings('ignore')


class RealTimeAnalyzer:
    """Real-time Elliott Wave analysis system."""
    
    def __init__(self, symbols=None, update_interval=300):  # 5 minutes
        """Initialize real-time analyzer.
        
        Args:
            symbols: List of symbols to monitor
            update_interval: Update interval in seconds
        """
        self.symbols = symbols or ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
        self.update_interval = update_interval
        
        self.data_loader = DataLoader()
        self.wave_detector = WaveDetector()
        self.fib_analyzer = FibonacciAnalyzer()
        self.strategy = ElliottWaveStrategy()
        self.visualizer = WaveVisualizer()
        
        self.last_analysis = {}
        self.alerts = []
        
    def analyze_symbol(self, symbol):
        """Analyze a single symbol."""
        try:
            # Load recent data
            data = self.data_loader.get_yahoo_data(symbol, period="3mo")
            current_price = data['close'].iloc[-1]
            
            # Detect waves
            waves = self.wave_detector.detect_waves(data)
            
            # Generate signals
            signals = self.strategy.generate_signals(data, symbol)
            
            # Get current wave analysis
            current_analysis = self.wave_detector.get_current_wave_count(data)
            
            # Calculate key levels
            recent_high = data['high'].tail(50).max()
            recent_low = data['low'].tail(50).min()
            
            fib_analysis = None
            if recent_high != recent_low:
                fib_analysis = self.fib_analyzer.analyze_retracement(
                    recent_high, recent_low, current_price, 'up'
                )
            
            # Check for alerts
            self.check_for_alerts(symbol, {
                'price': current_price,
                'signals': signals,
                'current_analysis': current_analysis,
                'waves': waves,
                'fib_analysis': fib_analysis
            })
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'price': current_price,
                'signals': signals,
                'waves': waves,
                'current_analysis': current_analysis,
                'fib_analysis': fib_analysis,
                'recent_high': recent_high,
                'recent_low': recent_low
            }
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None
    
    def check_for_alerts(self, symbol, analysis):
        """Check for trading alerts."""
        alerts = []
        
        # High confidence signal alert
        signals = analysis.get('signals', [])
        if signals:
            latest_signal = signals[-1]
            signal_age = (datetime.now() - latest_signal.timestamp).total_seconds() / 3600  # hours
            
            if latest_signal.confidence > 0.8 and signal_age < 24:  # Recent high-confidence signal
                alerts.append({
                    'type': 'HIGH_CONFIDENCE_SIGNAL',
                    'symbol': symbol,
                    'message': f"{symbol}: High confidence {latest_signal.signal_type.value} signal "
                              f"(confidence: {latest_signal.confidence:.2f})",
                    'timestamp': datetime.now(),
                    'signal': latest_signal
                })
        
        # Wave completion alert
        waves = analysis.get('waves', [])
        if waves and len(waves) >= 2:
            latest_wave = waves[-1]
            previous_wave = waves[-2]
            
            # Check if we just completed a significant wave
            wave_completion_threshold = 0.75
            if (latest_wave.confidence > wave_completion_threshold and 
                latest_wave.wave_type != previous_wave.wave_type):
                
                alerts.append({
                    'type': 'WAVE_COMPLETION',
                    'symbol': symbol,
                    'message': f"{symbol}: Completed {latest_wave.wave_type.value} wave "
                              f"(confidence: {latest_wave.confidence:.2f})",
                    'timestamp': datetime.now(),
                    'wave': latest_wave
                })
        
        # Fibonacci level alert
        fib_analysis = analysis.get('fib_analysis')
        if fib_analysis:
            price = analysis['price']
            price_analysis = self.fib_analyzer.analyze_price_at_fibonacci(price, fib_analysis)
            
            # Check if price is near a significant Fibonacci level
            if price_analysis.get('proximity_to_level', 1.0) < 0.02:  # Within 2%
                level = price_analysis.get('closest_level')
                if level:
                    alerts.append({
                        'type': 'FIBONACCI_LEVEL',
                        'symbol': symbol,
                        'message': f"{symbol}: Near Fibonacci {level.ratio:.1%} level "
                                  f"at ${level.price:.2f}",
                        'timestamp': datetime.now(),
                        'level': level
                    })
        
        # Add alerts to global list
        self.alerts.extend(alerts)
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]
    
    def run_analysis_cycle(self):
        """Run one complete analysis cycle."""
        print(f"\n{'='*60}")
        print(f"Real-time Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        results = {}
        
        for symbol in self.symbols:
            print(f"\nAnalyzing {symbol}...")
            result = self.analyze_symbol(symbol)
            
            if result:
                results[symbol] = result
                
                # Display summary
                price = result['price']
                signals = result['signals']
                current_analysis = result['current_analysis']
                
                print(f"  Price: ${price:.2f}")
                print(f"  Current Wave: {current_analysis['current_wave']}")
                print(f"  Confidence: {current_analysis['confidence']:.2f}")
                
                if signals:
                    latest_signal = signals[-1]
                    signal_age = (datetime.now() - latest_signal.timestamp).total_seconds() / 3600
                    print(f"  Latest Signal: {latest_signal.signal_type.value} "
                          f"({signal_age:.1f}h ago, conf: {latest_signal.confidence:.2f})")
        
        # Display alerts
        recent_alerts = [alert for alert in self.alerts 
                        if (datetime.now() - alert['timestamp']).total_seconds() < 3600]  # Last hour
        
        if recent_alerts:
            print(f"\n{'='*30}")
            print("RECENT ALERTS")
            print(f"{'='*30}")
            
            for alert in recent_alerts[-5:]:  # Last 5 alerts
                age_minutes = (datetime.now() - alert['timestamp']).total_seconds() / 60
                print(f"[{age_minutes:.0f}m ago] {alert['message']}")
        
        return results
    
    def start_monitoring(self, cycles=None):
        """Start real-time monitoring.
        
        Args:
            cycles: Number of cycles to run (None for infinite)
        """
        print(f"Starting real-time Elliott Wave monitoring...")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Update interval: {self.update_interval} seconds")
        print(f"Press Ctrl+C to stop")
        
        cycle_count = 0
        
        try:
            while cycles is None or cycle_count < cycles:
                results = self.run_analysis_cycle()
                
                # Store last analysis for comparison
                self.last_analysis = results
                
                cycle_count += 1
                
                if cycles is None or cycle_count < cycles:
                    print(f"\nWaiting {self.update_interval} seconds for next update...")
                    time.sleep(self.update_interval)
        
        except KeyboardInterrupt:
            print(f"\nMonitoring stopped by user.")
        
        print(f"\nCompleted {cycle_count} analysis cycles.")
    
    def generate_dashboard_data(self):
        """Generate data for a trading dashboard."""
        dashboard_data = {
            'timestamp': datetime.now(),
            'symbols': {},
            'alerts': self.alerts[-10:],  # Last 10 alerts
            'summary': {
                'total_symbols': len(self.symbols),
                'active_alerts': len([a for a in self.alerts 
                                    if (datetime.now() - a['timestamp']).total_seconds() < 3600]),
                'high_confidence_signals': 0
            }
        }
        
        for symbol in self.symbols:
            result = self.analyze_symbol(symbol)
            if result:
                signals = result['signals']
                high_conf_signals = [s for s in signals if s.confidence > 0.7]
                
                dashboard_data['symbols'][symbol] = {
                    'price': result['price'],
                    'current_wave': result['current_analysis']['current_wave'],
                    'confidence': result['current_analysis']['confidence'],
                    'signals_count': len(signals),
                    'high_confidence_signals': len(high_conf_signals),
                    'last_signal': signals[-1].signal_type.value if signals else None,
                    'trend': self.determine_trend(result)
                }
                
                dashboard_data['summary']['high_confidence_signals'] += len(high_conf_signals)
        
        return dashboard_data
    
    def determine_trend(self, result):
        """Determine overall trend for a symbol."""
        data = self.data_loader.get_yahoo_data(result['symbol'], period="1mo")
        
        if len(data) < 20:
            return "UNKNOWN"
        
        # Simple trend analysis
        recent_prices = data['close'].tail(20)
        price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        
        if price_change > 0.05:  # 5% gain
            return "UPTREND"
        elif price_change < -0.05:  # 5% loss
            return "DOWNTREND"
        else:
            return "SIDEWAYS"


def demo_realtime_analysis():
    """Demonstrate real-time analysis with a few cycles."""
    print("=== Real-time Elliott Wave Analysis Demo ===")
    
    # Create analyzer with shorter update interval for demo
    analyzer = RealTimeAnalyzer(
        symbols=["AAPL", "MSFT", "TSLA"],
        update_interval=10  # 10 seconds for demo
    )
    
    # Run 3 analysis cycles
    analyzer.start_monitoring(cycles=3)
    
    # Generate dashboard data
    print(f"\n{'='*40}")
    print("DASHBOARD DATA")
    print(f"{'='*40}")
    
    dashboard = analyzer.generate_dashboard_data()
    
    print(f"Generated at: {dashboard['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total symbols: {dashboard['summary']['total_symbols']}")
    print(f"Active alerts: {dashboard['summary']['active_alerts']}")
    print(f"High confidence signals: {dashboard['summary']['high_confidence_signals']}")
    
    print(f"\nSymbol Overview:")
    print(f"{'Symbol':<8} {'Price':<10} {'Wave':<8} {'Trend':<10} {'Signals':<8}")
    print("-" * 55)
    
    for symbol, data in dashboard['symbols'].items():
        print(f"{symbol:<8} ${data['price']:<9.2f} {data['current_wave']:<8} "
              f"{data['trend']:<10} {data['signals_count']:<8}")


def demo_alert_system():
    """Demonstrate the alert system."""
    print(f"\n{'='*40}")
    print("ALERT SYSTEM DEMO")
    print(f"{'='*40}")
    
    analyzer = RealTimeAnalyzer(symbols=["NVDA"])
    
    # Run analysis to generate potential alerts
    result = analyzer.analyze_symbol("NVDA")
    
    if analyzer.alerts:
        print(f"Generated {len(analyzer.alerts)} alerts:")
        for alert in analyzer.alerts:
            print(f"  [{alert['type']}] {alert['message']}")
    else:
        print("No alerts generated in this analysis.")
    
    # Simulate some historical alerts for demo
    demo_alerts = [
        {
            'type': 'HIGH_CONFIDENCE_SIGNAL',
            'symbol': 'NVDA',
            'message': 'NVDA: High confidence BUY signal (confidence: 0.85)',
            'timestamp': datetime.now() - timedelta(hours=2)
        },
        {
            'type': 'WAVE_COMPLETION',
            'symbol': 'NVDA',
            'message': 'NVDA: Completed Wave 3 (confidence: 0.78)',
            'timestamp': datetime.now() - timedelta(hours=4)
        },
        {
            'type': 'FIBONACCI_LEVEL',
            'symbol': 'NVDA',
            'message': 'NVDA: Near Fibonacci 61.8% level at $850.50',
            'timestamp': datetime.now() - timedelta(hours=1)
        }
    ]
    
    print(f"\nDemo Historical Alerts:")
    for alert in demo_alerts:
        age_hours = (datetime.now() - alert['timestamp']).total_seconds() / 3600
        print(f"  [{age_hours:.1f}h ago] {alert['message']}")


if __name__ == "__main__":
    demo_realtime_analysis()
    demo_alert_system()
    
    print(f"\n{'='*60}")
    print("Real-time analysis demo completed!")
    print("To run continuous monitoring, modify the script to call:")
    print("analyzer.start_monitoring()  # Runs indefinitely")
    print(f"{'='*60}")
