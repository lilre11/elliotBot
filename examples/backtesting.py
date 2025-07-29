"""
Example: Strategy Backtesting
This example demonstrates how to backtest Elliott Wave trading strategies.
"""

import sys
import os
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.data_loader import DataLoader
from src.trading.strategy import ElliottWaveStrategy
from src.trading.backtester import BacktestEngine

warnings.filterwarnings('ignore')


def main():
    print("=== Elliott Wave Strategy Backtesting Example ===")
    
    # Load data for multiple symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    period = "2y"
    
    loader = DataLoader()
    backtester = BacktestEngine()
    strategy = ElliottWaveStrategy()
    
    print(f"Testing symbols: {symbols}")
    print(f"Period: {period}")
    print(f"Initial Capital: ${backtester.initial_capital:,.2f}")
    
    results_summary = []
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"BACKTESTING {symbol}")
        print(f"{'='*50}")
        
        try:
            # Load data
            print(f"Loading {symbol} data...")
            data = loader.get_yahoo_data(symbol, period=period)
            print(f"Loaded {len(data)} data points")
            
            # Run backtest
            print("Running backtest...")
            results = backtester.run_backtest(strategy, data, symbol)
            
            # Generate report
            report = backtester.generate_report(results)
            results_summary.append({
                'symbol': symbol,
                'report': report
            })
            
            # Display results
            print(f"\nResults for {symbol}:")
            print(f"Total Return: {report['summary']['total_return_pct']:.2f}%")
            print(f"Annual Return: {report['summary']['annual_return_pct']:.2f}%")
            print(f"Max Drawdown: {report['summary']['max_drawdown_pct']:.2f}%")
            print(f"Sharpe Ratio: {report['summary']['sharpe_ratio']:.2f}")
            print(f"Win Rate: {report['trade_stats']['win_rate']:.2f}%")
            print(f"Total Trades: {report['trade_stats']['total_trades']}")
            print(f"Profit Factor: {report['summary']['profit_factor']:.2f}")
            
            # Show best trades
            if results.trades:
                profitable_trades = [t for t in results.trades if t.pnl > 0]
                if profitable_trades:
                    best_trade = max(profitable_trades, key=lambda x: x.pnl)
                    print(f"Best Trade: {best_trade.side} {best_trade.symbol} "
                          f"P&L: ${best_trade.pnl:.2f} ({best_trade.pnl_pct:.2%}) "
                          f"Wave: {best_trade.wave_type}")
        
        except Exception as e:
            print(f"Error backtesting {symbol}: {e}")
            continue
    
    # Summary comparison
    if results_summary:
        print(f"\n{'='*60}")
        print("SUMMARY COMPARISON")
        print(f"{'='*60}")
        
        print(f"{'Symbol':<10} {'Return %':<10} {'Annual %':<10} {'Sharpe':<8} {'Max DD %':<10} {'Win Rate %':<12} {'Trades':<8}")
        print("-" * 60)
        
        for result in results_summary:
            symbol = result['symbol']
            r = result['report']['summary']
            t = result['report']['trade_stats']
            
            print(f"{symbol:<10} {r['total_return_pct']:<10.2f} {r['annual_return_pct']:<10.2f} "
                  f"{r['sharpe_ratio']:<8.2f} {r['max_drawdown_pct']:<10.2f} "
                  f"{t['win_rate']:<12.2f} {t['total_trades']:<8}")
        
        # Best performer
        best_performer = max(results_summary, key=lambda x: x['report']['summary']['sharpe_ratio'])
        print(f"\nBest Performer (by Sharpe Ratio): {best_performer['symbol']}")
        
        # Calculate portfolio performance if equally weighted
        avg_return = sum(r['report']['summary']['total_return_pct'] for r in results_summary) / len(results_summary)
        avg_sharpe = sum(r['report']['summary']['sharpe_ratio'] for r in results_summary) / len(results_summary)
        avg_max_dd = sum(r['report']['summary']['max_drawdown_pct'] for r in results_summary) / len(results_summary)
        
        print(f"\nPortfolio Average (Equal Weight):")
        print(f"Average Return: {avg_return:.2f}%")
        print(f"Average Sharpe: {avg_sharpe:.2f}")
        print(f"Average Max DD: {avg_max_dd:.2f}%")
    
    print("\nBacktesting example completed!")


def backtest_single_symbol_detailed():
    """Detailed backtest example for a single symbol."""
    print("\n" + "="*60)
    print("DETAILED SINGLE SYMBOL BACKTEST")
    print("="*60)
    
    symbol = "AAPL"
    period = "2y"
    
    # Load data
    loader = DataLoader()
    data = loader.get_yahoo_data(symbol, period=period)
    
    # Initialize components
    strategy = ElliottWaveStrategy()
    backtester = BacktestEngine()
    
    # Run backtest
    results = backtester.run_backtest(strategy, data, symbol)
    
    # Detailed analysis
    print(f"\nDetailed Analysis for {symbol}:")
    print(f"Backtest Period: {results.start_date.strftime('%Y-%m-%d')} to {results.end_date.strftime('%Y-%m-%d')}")
    print(f"Total Duration: {(results.end_date - results.start_date).days} days")
    
    print(f"\nCapital Evolution:")
    print(f"Initial: ${results.initial_capital:,.2f}")
    print(f"Final: ${results.final_capital:,.2f}")
    print(f"Peak: ${results.peak_capital:,.2f}")
    print(f"Total Return: ${results.total_return:,.2f} ({results.total_return_pct:.2%})")
    print(f"Annualized Return: {results.annual_return_pct:.2%}")
    
    print(f"\nRisk Metrics:")
    print(f"Maximum Drawdown: ${results.max_drawdown:,.2f} ({results.max_drawdown_pct:.2%})")
    print(f"Volatility: {results.volatility:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    
    print(f"\nTrading Statistics:")
    print(f"Total Trades: {results.total_trades}")
    print(f"Winning Trades: {results.winning_trades}")
    print(f"Losing Trades: {results.losing_trades}")
    print(f"Win Rate: {results.win_rate:.2%}")
    print(f"Average Win: ${results.avg_win:.2f}")
    print(f"Average Loss: ${results.avg_loss:.2f}")
    print(f"Largest Win: ${results.largest_win:.2f}")
    print(f"Largest Loss: ${results.largest_loss:.2f}")
    print(f"Profit Factor: {results.profit_factor:.2f}")
    print(f"Average Holding Period: {results.avg_holding_period_hours:.1f} hours")
    
    # Trade analysis by wave type
    if results.trades:
        print(f"\nTrade Analysis by Wave Type:")
        wave_stats = {}
        for trade in results.trades:
            wave_type = trade.wave_type or "Unknown"
            if wave_type not in wave_stats:
                wave_stats[wave_type] = {'count': 0, 'total_pnl': 0, 'wins': 0}
            
            wave_stats[wave_type]['count'] += 1
            wave_stats[wave_type]['total_pnl'] += trade.pnl
            if trade.pnl > 0:
                wave_stats[wave_type]['wins'] += 1
        
        for wave_type, stats in wave_stats.items():
            win_rate = stats['wins'] / stats['count'] if stats['count'] > 0 else 0
            avg_pnl = stats['total_pnl'] / stats['count'] if stats['count'] > 0 else 0
            print(f"  {wave_type}: {stats['count']} trades, "
                  f"Win Rate: {win_rate:.2%}, "
                  f"Avg P&L: ${avg_pnl:.2f}")
    
    print(f"\nDetailed backtest completed!")


if __name__ == "__main__":
    main()
    backtest_single_symbol_detailed()
