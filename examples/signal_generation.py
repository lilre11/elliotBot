"""
Example: Signal Generation
This example demonstrates how to generate trading signals using Elliott Wave analysis.
"""

import sys
import os
import warnings
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import DataLoader
from src.trading.strategy import ElliottWaveStrategy
from src.analysis.wave_detector import WaveDetector
from src.analysis.fibonacci import FibonacciAnalyzer

warnings.filterwarnings('ignore')


def main():
    print("=== Elliott Wave Signal Generation Example ===")
    
    # Symbols to analyze
    symbols = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN"]
    period = "6mo"
    
    loader = DataLoader()
    strategy = ElliottWaveStrategy()
    
    print(f"Analyzing symbols: {symbols}")
    print(f"Analysis period: {period}")
    print(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")
    
    all_signals = []
    
    for symbol in symbols:
        print(f"\n{'='*40}")
        print(f"ANALYZING {symbol}")
        print(f"{'='*40}")
        
        try:
            # Load data
            data = loader.get_yahoo_data(symbol, period=period)
            current_price = data['close'].iloc[-1]
            
            print(f"Current Price: ${current_price:.2f}")
            print(f"Data points: {len(data)}")
            
            # Generate signals
            signals = strategy.generate_signals(data, symbol)
            
            if signals:
                print(f"\nGenerated {len(signals)} signals:")
                
                for i, signal in enumerate(signals):
                    days_ago = (data.index[-1] - signal.timestamp).days
                    
                    print(f"\n  Signal {i+1}:")
                    print(f"    Type: {signal.signal_type.value}")
                    print(f"    Date: {signal.timestamp.strftime('%Y-%m-%d')} ({days_ago} days ago)")
                    print(f"    Price: ${signal.price:.2f}")
                    print(f"    Current Price: ${current_price:.2f}")
                    print(f"    Confidence: {signal.confidence:.2f}")
                    print(f"    Wave Type: {signal.wave_type}")
                    
                    if signal.fibonacci_level:
                        print(f"    Fibonacci Level: {signal.fibonacci_level:.1%}")
                    
                    if signal.stop_loss:
                        print(f"    Stop Loss: ${signal.stop_loss:.2f}")
                        risk_pct = abs(signal.price - signal.stop_loss) / signal.price * 100
                        print(f"    Risk: {risk_pct:.1f}%")
                    
                    if signal.take_profit:
                        print(f"    Take Profit: ${signal.take_profit:.2f}")
                        reward_pct = abs(signal.take_profit - signal.price) / signal.price * 100
                        print(f"    Reward: {reward_pct:.1f}%")
                        
                        if signal.stop_loss:
                            risk = abs(signal.price - signal.stop_loss)
                            reward = abs(signal.take_profit - signal.price)
                            rr_ratio = reward / risk if risk > 0 else 0
                            print(f"    Risk/Reward Ratio: 1:{rr_ratio:.1f}")
                    
                    print(f"    Reason: {signal.reason}")
                    
                    # Calculate current performance if signal was taken
                    if signal.signal_type.value in ['BUY', 'SELL']:
                        if signal.signal_type.value == 'BUY':
                            current_pnl_pct = (current_price - signal.price) / signal.price * 100
                        else:  # SELL
                            current_pnl_pct = (signal.price - current_price) / signal.price * 100
                        
                        print(f"    Current P&L: {current_pnl_pct:+.1f}%")
                
                # Add to all signals list
                all_signals.extend([(symbol, signal) for signal in signals])
            
            else:
                print("No signals generated.")
            
            # Get current wave analysis
            detector = WaveDetector()
            current_analysis = detector.get_current_wave_count(data)
            
            print(f"\nCurrent Wave Analysis:")
            print(f"  Current Wave: {current_analysis['current_wave']}")
            print(f"  Confidence: {current_analysis['confidence']:.2f}")
            print(f"  Analysis: {current_analysis['analysis']}")
            
            if current_analysis.get('next_target'):
                next_target = current_analysis['next_target']
                print(f"  Next Expected: {next_target.get('type', 'Unknown')}")
        
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            continue
    
    # Summary of all signals
    print(f"\n{'='*60}")
    print("SIGNAL SUMMARY")
    print(f"{'='*60}")
    
    if all_signals:
        # Recent signals (last 30 days)
        recent_signals = []
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for symbol, signal in all_signals:
            if signal.timestamp >= cutoff_date:
                recent_signals.append((symbol, signal))
        
        print(f"Total signals generated: {len(all_signals)}")
        print(f"Recent signals (last 30 days): {len(recent_signals)}")
        
        if recent_signals:
            print(f"\nRecent High-Confidence Signals:")
            print(f"{'Symbol':<8} {'Type':<5} {'Date':<12} {'Price':<10} {'Confidence':<10} {'Wave':<6}")
            print("-" * 65)
            
            # Sort by confidence
            recent_signals.sort(key=lambda x: x[1].confidence, reverse=True)
            
            for symbol, signal in recent_signals[:10]:  # Top 10
                print(f"{symbol:<8} {signal.signal_type.value:<5} "
                      f"{signal.timestamp.strftime('%Y-%m-%d'):<12} "
                      f"${signal.price:<9.2f} {signal.confidence:<10.2f} {signal.wave_type or 'N/A':<6}")
        
        # Signal type distribution
        signal_types = {}
        for symbol, signal in all_signals:
            sig_type = signal.signal_type.value
            if sig_type not in signal_types:
                signal_types[sig_type] = 0
            signal_types[sig_type] += 1
        
        print(f"\nSignal Type Distribution:")
        for sig_type, count in signal_types.items():
            print(f"  {sig_type}: {count}")
        
        # Wave type distribution
        wave_types = {}
        for symbol, signal in all_signals:
            wave_type = signal.wave_type or "Unknown"
            if wave_type not in wave_types:
                wave_types[wave_type] = 0
            wave_types[wave_type] += 1
        
        print(f"\nWave Type Distribution:")
        for wave_type, count in wave_types.items():
            print(f"  {wave_type}: {count}")
        
        # Average confidence by wave type
        wave_confidence = {}
        for symbol, signal in all_signals:
            wave_type = signal.wave_type or "Unknown"
            if wave_type not in wave_confidence:
                wave_confidence[wave_type] = []
            wave_confidence[wave_type].append(signal.confidence)
        
        print(f"\nAverage Confidence by Wave Type:")
        for wave_type, confidences in wave_confidence.items():
            avg_conf = sum(confidences) / len(confidences)
            print(f"  {wave_type}: {avg_conf:.2f}")
    
    else:
        print("No signals generated for any symbol.")
    
    print(f"\nSignal generation example completed!")


def analyze_specific_opportunity():
    """Analyze a specific trading opportunity in detail."""
    print(f"\n{'='*60}")
    print("SPECIFIC OPPORTUNITY ANALYSIS")
    print(f"{'='*60}")
    
    symbol = "TSLA"
    period = "1y"
    
    loader = DataLoader()
    data = loader.get_yahoo_data(symbol, period=period)
    
    strategy = ElliottWaveStrategy()
    signals = strategy.generate_signals(data, symbol)
    
    detector = WaveDetector()
    waves = detector.detect_waves(data)
    
    fib_analyzer = FibonacciAnalyzer()
    
    current_price = data['close'].iloc[-1]
    
    print(f"Detailed Analysis for {symbol}")
    print(f"Current Price: ${current_price:.2f}")
    
    # Recent price action
    recent_high = data['high'].tail(20).max()
    recent_low = data['low'].tail(20).min()
    price_range = (current_price - recent_low) / (recent_high - recent_low)
    
    print(f"Recent High: ${recent_high:.2f}")
    print(f"Recent Low: ${recent_low:.2f}")
    print(f"Position in Range: {price_range:.1%}")
    
    # Fibonacci analysis
    if recent_high != recent_low:
        fib_analysis = fib_analyzer.analyze_retracement(
            recent_high, recent_low, current_price, 'up'
        )
        
        price_analysis = fib_analyzer.analyze_price_at_fibonacci(current_price, fib_analysis)
        print(f"Fibonacci Position: {price_analysis['price_position']}")
        
        if price_analysis['next_support']:
            support = price_analysis['next_support']
            print(f"Next Support: ${support.price:.2f} ({support.ratio:.1%})")
        
        if price_analysis['next_resistance']:
            resistance = price_analysis['next_resistance']
            print(f"Next Resistance: ${resistance.price:.2f} ({resistance.ratio:.1%})")
    
    # Wave analysis
    if waves:
        latest_wave = waves[-1]
        print(f"Latest Wave: {latest_wave.wave_type.value}")
        print(f"Wave Confidence: {latest_wave.confidence:.2f}")
        print(f"Wave Direction: {latest_wave.direction.name}")
    
    # Signal analysis
    if signals:
        latest_signal = signals[-1]
        days_since = (data.index[-1] - latest_signal.timestamp).days
        
        print(f"\nLatest Signal:")
        print(f"  Type: {latest_signal.signal_type.value}")
        print(f"  Days Since: {days_since}")
        print(f"  Confidence: {latest_signal.confidence:.2f}")
        print(f"  Reason: {latest_signal.reason}")
        
        if latest_signal.stop_loss and latest_signal.take_profit:
            risk = abs(current_price - latest_signal.stop_loss)
            reward = abs(latest_signal.take_profit - current_price)
            print(f"  Risk: ${risk:.2f}")
            print(f"  Reward: ${reward:.2f}")
            print(f"  Risk/Reward: 1:{reward/risk:.1f}" if risk > 0 else "  Risk/Reward: N/A")
    
    # Trading recommendation
    print(f"\nTrading Recommendation:")
    if signals and signals[-1].confidence > 0.7:
        latest_signal = signals[-1]
        print(f"  Action: {latest_signal.signal_type.value}")
        print(f"  Confidence: HIGH ({latest_signal.confidence:.2f})")
        print(f"  Basis: {latest_signal.reason}")
    else:
        print(f"  Action: WAIT")
        print(f"  Reason: Low confidence or no clear signals")
    
    print(f"\nSpecific opportunity analysis completed!")


if __name__ == "__main__":
    main()
    analyze_specific_opportunity()
