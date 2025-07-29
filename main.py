"""
Main entry point for Elliott Wave Trading Bot.
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_loader import DataLoader
from src.analysis.wave_detector import WaveDetector
from src.analysis.fibonacci import FibonacciAnalyzer
from src.visualization.visualizer import WaveVisualizer
from src.trading.strategy import ElliottWaveStrategy
from src.trading.backtester import BacktestEngine
from src.utils.config import get_config
from src.utils.logger import get_logger, setup_logging

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
setup_logging()
logger = get_logger(__name__)


def analyze_symbol(symbol: str, period: str = "1y", show_chart: bool = True, save_chart: bool = False):
    """
    Perform Elliott Wave analysis on a symbol.
    
    Args:
        symbol: Trading symbol to analyze
        period: Time period for analysis
        show_chart: Whether to display interactive chart
        save_chart: Whether to save chart to file
    """
    try:
        logger.info(f"Starting Elliott Wave analysis for {symbol}")
        
        # Load data
        loader = DataLoader()
        data = loader.get_yahoo_data(symbol, period=period)
        logger.info(f"Loaded {len(data)} data points for {symbol}")
        
        # Detect Elliott Waves
        detector = WaveDetector()
        waves = detector.detect_waves(data)
        logger.info(f"Detected {len(waves)} Elliott Waves")
        
        # Fibonacci analysis
        fib_analyzer = FibonacciAnalyzer()
        if len(data) > 50:
            high_price = data['high'].rolling(50).max().iloc[-1]
            low_price = data['low'].rolling(50).min().iloc[-1]
            current_price = data['close'].iloc[-1]
            
            fib_analysis = fib_analyzer.analyze_retracement(high_price, low_price, current_price, 'up')
        else:
            fib_analysis = None
        
        # Get current wave analysis
        current_analysis = detector.get_current_wave_count(data)
        
        # Print analysis results
        print(f"\n=== ELLIOTT WAVE ANALYSIS FOR {symbol} ===")
        print(f"Analysis Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Total Waves Detected: {len(waves)}")
        
        print(f"\n=== CURRENT WAVE STATUS ===")
        print(f"Current Wave: {current_analysis['current_wave']}")
        print(f"Confidence: {current_analysis['confidence']:.2f}")
        print(f"Analysis: {current_analysis['analysis']}")
        
        if current_analysis.get('next_target'):
            next_target = current_analysis['next_target']
            print(f"Next Expected: {next_target.get('type', 'Unknown')}")
            print(f"Direction: {next_target.get('direction', 'Unknown')}")
        
        # Print wave details
        if waves:
            print(f"\n=== DETECTED WAVES ===")
            for i, wave in enumerate(waves[-10:]):  # Show last 10 waves
                duration = (wave.end_point.timestamp - wave.start_point.timestamp).days
                price_change = ((wave.end_point.price - wave.start_point.price) / wave.start_point.price) * 100
                
                print(f"Wave {wave.wave_type.value}: "
                      f"{wave.start_point.timestamp.strftime('%Y-%m-%d')} -> "
                      f"{wave.end_point.timestamp.strftime('%Y-%m-%d')} "
                      f"({duration} days), "
                      f"Price: ${wave.start_point.price:.2f} -> ${wave.end_point.price:.2f} "
                      f"({price_change:+.1f}%), "
                      f"Confidence: {wave.confidence:.2f}")
        
        # Fibonacci analysis
        if fib_analysis:
            print(f"\n=== FIBONACCI ANALYSIS ===")
            print(f"Swing High: ${fib_analysis.swing_high:.2f}")
            print(f"Swing Low: ${fib_analysis.swing_low:.2f}")
            print(f"Trend Direction: {fib_analysis.trend_direction}")
            
            price_analysis = fib_analyzer.analyze_price_at_fibonacci(current_price, fib_analysis)
            print(f"Price Position: {price_analysis['price_position']}")
            
            if price_analysis['next_support']:
                print(f"Next Support: ${price_analysis['next_support'].price:.2f} ({price_analysis['next_support'].ratio:.1%})")
            if price_analysis['next_resistance']:
                print(f"Next Resistance: ${price_analysis['next_resistance'].price:.2f} ({price_analysis['next_resistance'].ratio:.1%})")
        
        # Visualization
        if show_chart or save_chart:
            visualizer = WaveVisualizer()
            
            # Create main chart
            fig = visualizer.plot_waves(data, waves, fib_analysis, title=f"Elliott Wave Analysis - {symbol}")
            
            if save_chart:
                save_path = f"charts/{symbol}_elliott_wave_analysis.html"
                Path("charts").mkdir(exist_ok=True)
                visualizer.save_chart(fig, save_path)
                print(f"\nChart saved to: {save_path}")
            
            if show_chart:
                fig.show()
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        print(f"Error: {e}")


def run_backtest(symbol: str, period: str = "2y", initial_capital: float = 100000):
    """
    Run backtest for Elliott Wave strategy.
    
    Args:
        symbol: Trading symbol
        period: Time period for backtest
        initial_capital: Starting capital
    """
    try:
        logger.info(f"Starting backtest for {symbol}")
        
        # Load data
        loader = DataLoader()
        data = loader.get_yahoo_data(symbol, period=period)
        
        # Initialize strategy and backtester
        strategy = ElliottWaveStrategy()
        backtester = BacktestEngine()
        backtester.initial_capital = initial_capital
        
        # Run backtest
        results = backtester.run_backtest(strategy, data, symbol)
        
        # Generate and display report
        report = backtester.generate_report(results)
        
        print(f"\n=== BACKTEST RESULTS FOR {symbol} ===")
        print(f"Period: {report['period']['start_date']} to {report['period']['end_date']} ({report['period']['duration_days']} days)")
        
        print(f"\n=== PERFORMANCE SUMMARY ===")
        print(f"Initial Capital: ${report['summary']['initial_capital']:,.2f}")
        print(f"Final Capital: ${report['summary']['final_capital']:,.2f}")
        print(f"Total Return: ${report['summary']['total_return']:,.2f} ({report['summary']['total_return_pct']:.2f}%)")
        print(f"Annualized Return: {report['summary']['annual_return_pct']:.2f}%")
        print(f"Maximum Drawdown: {report['summary']['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio: {report['summary']['sharpe_ratio']:.2f}")
        print(f"Volatility: {report['summary']['volatility']:.2f}")
        print(f"Profit Factor: {report['summary']['profit_factor']:.2f}")
        
        print(f"\n=== TRADE STATISTICS ===")
        print(f"Total Trades: {report['trade_stats']['total_trades']}")
        print(f"Winning Trades: {report['trade_stats']['winning_trades']}")
        print(f"Losing Trades: {report['trade_stats']['losing_trades']}")
        print(f"Win Rate: {report['trade_stats']['win_rate']:.2f}%")
        print(f"Average Win: ${report['trade_stats']['avg_win']:.2f}")
        print(f"Average Loss: ${report['trade_stats']['avg_loss']:.2f}")
        print(f"Largest Win: ${report['trade_stats']['largest_win']:.2f}")
        print(f"Largest Loss: ${report['trade_stats']['largest_loss']:.2f}")
        print(f"Average Holding Period: {report['trade_stats']['avg_holding_period_hours']:.1f} hours")
        
        # Show sample trades
        if results.trades:
            print(f"\n=== SAMPLE TRADES ===")
            for i, trade in enumerate(results.trades[:10]):  # Show first 10 trades
                print(f"Trade {i+1}: {trade.side} {trade.symbol} "
                      f"Entry: {trade.entry_time.strftime('%Y-%m-%d')} @ ${trade.entry_price:.2f} "
                      f"Exit: {trade.exit_time.strftime('%Y-%m-%d')} @ ${trade.exit_price:.2f} "
                      f"P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2%}) "
                      f"Wave: {trade.wave_type}")
        
        # Save results
        results_path = f"backtest_results_{symbol}_{period}.json"
        import json
        with open(results_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Error running backtest for {symbol}: {e}")
        print(f"Error: {e}")


def generate_signals(symbol: str, period: str = "6mo"):
    """
    Generate current trading signals for a symbol.
    
    Args:
        symbol: Trading symbol
        period: Time period for analysis
    """
    try:
        logger.info(f"Generating signals for {symbol}")
        
        # Load data
        loader = DataLoader()
        data = loader.get_yahoo_data(symbol, period=period)
        
        # Generate signals
        strategy = ElliottWaveStrategy()
        signals = strategy.generate_signals(data, symbol)
        
        current_price = data['close'].iloc[-1]
        
        print(f"\n=== TRADING SIGNALS FOR {symbol} ===")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Analysis Date: {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"Generated Signals: {len(signals)}")
        
        if signals:
            print(f"\n=== ACTIVE SIGNALS ===")
            for i, signal in enumerate(signals[-5:]):  # Show last 5 signals
                days_ago = (data.index[-1] - signal.timestamp).days
                
                print(f"\nSignal {i+1}:")
                print(f"  Date: {signal.timestamp.strftime('%Y-%m-%d')} ({days_ago} days ago)")
                print(f"  Type: {signal.signal_type.value}")
                print(f"  Price: ${signal.price:.2f}")
                print(f"  Confidence: {signal.confidence:.2f}")
                print(f"  Wave Type: {signal.wave_type}")
                if signal.fibonacci_level:
                    print(f"  Fibonacci Level: {signal.fibonacci_level:.1%}")
                if signal.stop_loss:
                    print(f"  Stop Loss: ${signal.stop_loss:.2f}")
                if signal.take_profit:
                    print(f"  Take Profit: ${signal.take_profit:.2f}")
                print(f"  Reason: {signal.reason}")
        else:
            print("No active signals found.")
        
    except Exception as e:
        logger.error(f"Error generating signals for {symbol}: {e}")
        print(f"Error: {e}")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Elliott Wave Trading Bot")
    parser.add_argument("command", choices=["analyze", "backtest", "signals"], 
                       help="Command to execute")
    parser.add_argument("symbol", help="Trading symbol (e.g., AAPL, EURUSD=X, BTC-USD)")
    parser.add_argument("--period", default="1y", 
                       help="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)")
    parser.add_argument("--capital", type=float, default=100000, 
                       help="Initial capital for backtesting")
    parser.add_argument("--no-chart", action="store_true", 
                       help="Don't display interactive charts")
    parser.add_argument("--save-chart", action="store_true", 
                       help="Save charts to files")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    if args.config:
        config = get_config(args.config)
    
    print(f"Elliott Wave Trading Bot")
    print(f"Command: {args.command}")
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.period}")
    
    try:
        if args.command == "analyze":
            analyze_symbol(
                symbol=args.symbol,
                period=args.period,
                show_chart=not args.no_chart,
                save_chart=args.save_chart
            )
        
        elif args.command == "backtest":
            run_backtest(
                symbol=args.symbol,
                period=args.period,
                initial_capital=args.capital
            )
        
        elif args.command == "signals":
            generate_signals(
                symbol=args.symbol,
                period=args.period
            )
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
