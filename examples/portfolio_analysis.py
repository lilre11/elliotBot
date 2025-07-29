"""
Example: Portfolio Analysis
This example demonstrates Elliott Wave analysis across a portfolio of assets.
"""

import sys
import os
import warnings
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.data_loader import DataLoader
from src.analysis.wave_detector import WaveDetector
from src.analysis.fibonacci import FibonacciAnalyzer
from src.trading.strategy import ElliottWaveStrategy
from src.trading.backtester import BacktestEngine

warnings.filterwarnings('ignore')


class PortfolioAnalyzer:
    """Portfolio-level Elliott Wave analysis."""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.wave_detector = WaveDetector()
        self.fib_analyzer = FibonacciAnalyzer()
        self.strategy = ElliottWaveStrategy()
        self.backtester = BacktestEngine()
    
    def analyze_portfolio(self, portfolio, period="1y"):
        """Analyze a portfolio of assets.
        
        Args:
            portfolio: Dict with symbol as key and weight as value
            period: Analysis period
            
        Returns:
            Portfolio analysis results
        """
        results = {
            'timestamp': datetime.now(),
            'portfolio': portfolio,
            'period': period,
            'symbols': {},
            'summary': {}
        }
        
        total_signals = 0
        total_high_conf_signals = 0
        portfolio_score = 0
        sector_analysis = {}
        
        print(f"Analyzing portfolio with {len(portfolio)} assets...")
        print(f"Period: {period}")
        
        for symbol, weight in portfolio.items():
            print(f"\nAnalyzing {symbol} (weight: {weight:.1%})...")
            
            try:
                # Load data
                data = self.data_loader.get_yahoo_data(symbol, period=period)
                current_price = data['close'].iloc[-1]
                
                # Detect waves
                waves = self.wave_detector.detect_waves(data)
                
                # Generate signals
                signals = self.strategy.generate_signals(data, symbol)
                
                # Current analysis
                current_analysis = self.wave_detector.get_current_wave_count(data)
                
                # Performance metrics
                price_change = self.calculate_price_change(data)
                volatility = self.calculate_volatility(data)
                
                # Fibonacci analysis
                recent_high = data['high'].tail(50).max()
                recent_low = data['low'].tail(50).min()
                fib_analysis = None
                
                if recent_high != recent_low:
                    fib_analysis = self.fib_analyzer.analyze_retracement(
                        recent_high, recent_low, current_price, 'up'
                    )
                
                # Calculate symbol score
                symbol_score = self.calculate_symbol_score(
                    signals, current_analysis, waves, price_change
                )
                
                # Store results
                symbol_result = {
                    'weight': weight,
                    'current_price': current_price,
                    'price_change': price_change,
                    'volatility': volatility,
                    'waves_count': len(waves),
                    'signals_count': len(signals),
                    'high_conf_signals': len([s for s in signals if s.confidence > 0.7]),
                    'current_wave': current_analysis['current_wave'],
                    'wave_confidence': current_analysis['confidence'],
                    'symbol_score': symbol_score,
                    'signals': signals,
                    'waves': waves,
                    'fib_analysis': fib_analysis
                }
                
                results['symbols'][symbol] = symbol_result
                
                # Update totals
                total_signals += len(signals)
                total_high_conf_signals += symbol_result['high_conf_signals']
                portfolio_score += symbol_score * weight
                
                # Sector analysis (simplified)
                sector = self.get_symbol_sector(symbol)
                if sector not in sector_analysis:
                    sector_analysis[sector] = {
                        'symbols': [],
                        'total_weight': 0,
                        'avg_score': 0,
                        'signal_count': 0
                    }
                
                sector_analysis[sector]['symbols'].append(symbol)
                sector_analysis[sector]['total_weight'] += weight
                sector_analysis[sector]['signal_count'] += len(signals)
                
                print(f"  Price: ${current_price:.2f} ({price_change:+.1%})")
                print(f"  Signals: {len(signals)} (high-conf: {symbol_result['high_conf_signals']})")
                print(f"  Current Wave: {current_analysis['current_wave']}")
                print(f"  Score: {symbol_score:.2f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        # Calculate sector averages
        for sector, data in sector_analysis.items():
            if data['symbols']:
                sector_scores = [results['symbols'][s]['symbol_score'] 
                               for s in data['symbols'] if s in results['symbols']]
                data['avg_score'] = sum(sector_scores) / len(sector_scores) if sector_scores else 0
        
        # Portfolio summary
        results['summary'] = {
            'total_symbols': len(portfolio),
            'analyzed_symbols': len(results['symbols']),
            'total_signals': total_signals,
            'high_conf_signals': total_high_conf_signals,
            'portfolio_score': portfolio_score,
            'sectors': sector_analysis,
            'top_opportunities': self.find_top_opportunities(results['symbols']),
            'risk_assessment': self.assess_portfolio_risk(results['symbols'])
        }
        
        return results
    
    def calculate_price_change(self, data, days=30):
        """Calculate price change over specified days."""
        if len(data) < days:
            days = len(data) - 1
        
        if days <= 0:
            return 0
        
        current_price = data['close'].iloc[-1]
        past_price = data['close'].iloc[-days-1]
        
        return (current_price - past_price) / past_price
    
    def calculate_volatility(self, data, days=30):
        """Calculate volatility over specified days."""
        if len(data) < days:
            days = len(data)
        
        returns = data['close'].tail(days).pct_change().dropna()
        return returns.std() * (252 ** 0.5)  # Annualized volatility
    
    def calculate_symbol_score(self, signals, current_analysis, waves, price_change):
        """Calculate a composite score for a symbol."""
        score = 0
        
        # Signal quality
        if signals:
            latest_signal = signals[-1]
            signal_age = (datetime.now() - latest_signal.timestamp).days
            
            if signal_age <= 7:  # Recent signal
                score += latest_signal.confidence * 30
            
            # High confidence signals
            high_conf_count = len([s for s in signals if s.confidence > 0.7])
            score += min(high_conf_count * 10, 30)
        
        # Wave analysis quality
        score += current_analysis['confidence'] * 20
        
        # Price momentum
        if abs(price_change) > 0.1:  # 10% move
            score += min(abs(price_change) * 100, 20)
        
        return min(score, 100)  # Cap at 100
    
    def get_symbol_sector(self, symbol):
        """Get sector for symbol (simplified mapping)."""
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        finance_symbols = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS']
        healthcare_symbols = ['JNJ', 'PFE', 'UNH', 'MRNA', 'ABBV']
        
        if symbol in tech_symbols:
            return 'Technology'
        elif symbol in finance_symbols:
            return 'Financial'
        elif symbol in healthcare_symbols:
            return 'Healthcare'
        else:
            return 'Other'
    
    def find_top_opportunities(self, symbols_data):
        """Find top trading opportunities in the portfolio."""
        opportunities = []
        
        for symbol, data in symbols_data.items():
            if data['signals']:
                latest_signal = data['signals'][-1]
                signal_age = (datetime.now() - latest_signal.timestamp).days
                
                if signal_age <= 14 and latest_signal.confidence > 0.6:  # Recent decent signal
                    opportunity = {
                        'symbol': symbol,
                        'signal_type': latest_signal.signal_type.value,
                        'confidence': latest_signal.confidence,
                        'days_old': signal_age,
                        'current_wave': data['current_wave'],
                        'symbol_score': data['symbol_score'],
                        'reason': latest_signal.reason
                    }
                    opportunities.append(opportunity)
        
        # Sort by confidence and recency
        opportunities.sort(
            key=lambda x: (x['confidence'], -x['days_old'], x['symbol_score']), 
            reverse=True
        )
        
        return opportunities[:5]  # Top 5
    
    def assess_portfolio_risk(self, symbols_data):
        """Assess overall portfolio risk."""
        risk_metrics = {
            'high_volatility_symbols': 0,
            'concentrated_positions': 0,
            'avg_volatility': 0,
            'correlation_risk': 'Medium',  # Simplified
            'sector_concentration': {}
        }
        
        volatilities = []
        
        for symbol, data in symbols_data.items():
            volatilities.append(data['volatility'])
            
            if data['volatility'] > 0.3:  # 30% annual volatility
                risk_metrics['high_volatility_symbols'] += 1
            
            if data['weight'] > 0.2:  # 20% position
                risk_metrics['concentrated_positions'] += 1
        
        if volatilities:
            risk_metrics['avg_volatility'] = sum(volatilities) / len(volatilities)
        
        return risk_metrics
    
    def generate_portfolio_report(self, results):
        """Generate a comprehensive portfolio report."""
        print(f"\n{'='*80}")
        print("ELLIOTT WAVE PORTFOLIO ANALYSIS REPORT")
        print(f"{'='*80}")
        
        summary = results['summary']
        
        print(f"Analysis Date: {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Period: {results['period']}")
        print(f"Portfolio Size: {summary['analyzed_symbols']}/{summary['total_symbols']} symbols")
        print(f"Portfolio Score: {summary['portfolio_score']:.1f}/100")
        
        # Performance overview
        print(f"\n{'='*50}")
        print("PERFORMANCE OVERVIEW")
        print(f"{'='*50}")
        
        total_weight = 0
        weighted_return = 0
        weighted_volatility = 0
        
        for symbol, data in results['symbols'].items():
            weight = data['weight']
            total_weight += weight
            weighted_return += data['price_change'] * weight
            weighted_volatility += data['volatility'] * weight
        
        print(f"Weighted Portfolio Return: {weighted_return:.1%}")
        print(f"Weighted Portfolio Volatility: {weighted_volatility:.1%}")
        print(f"Total Signals Generated: {summary['total_signals']}")
        print(f"High Confidence Signals: {summary['high_conf_signals']}")
        
        # Top performers
        print(f"\n{'='*50}")
        print("TOP PERFORMERS")
        print(f"{'='*50}")
        
        sorted_symbols = sorted(
            results['symbols'].items(),
            key=lambda x: x[1]['symbol_score'],
            reverse=True
        )
        
        print(f"{'Symbol':<8} {'Weight':<8} {'Return':<8} {'Score':<8} {'Wave':<10}")
        print("-" * 50)
        
        for symbol, data in sorted_symbols[:5]:
            print(f"{symbol:<8} {data['weight']:<7.1%} {data['price_change']:<7.1%} "
                  f"{data['symbol_score']:<7.1f} {data['current_wave']:<10}")
        
        # Trading opportunities
        print(f"\n{'='*50}")
        print("TRADING OPPORTUNITIES")
        print(f"{'='*50}")
        
        opportunities = summary['top_opportunities']
        if opportunities:
            print(f"{'Symbol':<8} {'Signal':<6} {'Conf':<6} {'Age':<5} {'Wave':<10}")
            print("-" * 40)
            
            for opp in opportunities:
                print(f"{opp['symbol']:<8} {opp['signal_type']:<6} "
                      f"{opp['confidence']:<5.2f} {opp['days_old']:<5}d {opp['current_wave']:<10}")
        else:
            print("No high-quality opportunities identified.")
        
        # Sector analysis
        print(f"\n{'='*50}")
        print("SECTOR ANALYSIS")
        print(f"{'='*50}")
        
        sectors = summary['sectors']
        print(f"{'Sector':<12} {'Weight':<8} {'Symbols':<8} {'Avg Score':<10} {'Signals':<8}")
        print("-" * 55)
        
        for sector, data in sectors.items():
            print(f"{sector:<12} {data['total_weight']:<7.1%} {len(data['symbols']):<8} "
                  f"{data['avg_score']:<9.1f} {data['signal_count']:<8}")
        
        # Risk assessment
        print(f"\n{'='*50}")
        print("RISK ASSESSMENT")
        print(f"{'='*50}")
        
        risk = summary['risk_assessment']
        print(f"Average Volatility: {risk['avg_volatility']:.1%}")
        print(f"High Volatility Symbols: {risk['high_volatility_symbols']}")
        print(f"Concentrated Positions (>20%): {risk['concentrated_positions']}")
        
        # Recommendations
        print(f"\n{'='*50}")
        print("RECOMMENDATIONS")
        print(f"{'='*50}")
        
        self.generate_recommendations(results)
    
    def generate_recommendations(self, results):
        """Generate portfolio recommendations."""
        summary = results['summary']
        
        recommendations = []
        
        # Portfolio score assessment
        if summary['portfolio_score'] > 70:
            recommendations.append("✓ Strong portfolio with good Elliott Wave opportunities")
        elif summary['portfolio_score'] > 50:
            recommendations.append("◐ Moderate portfolio strength, selective opportunities available")
        else:
            recommendations.append("⚠ Weak portfolio signals, consider defensive positioning")
        
        # Signal assessment
        if summary['high_conf_signals'] > 0:
            recommendations.append(f"✓ {summary['high_conf_signals']} high-confidence signals available")
        else:
            recommendations.append("◐ No high-confidence signals, wait for better setups")
        
        # Risk assessment
        risk = summary['risk_assessment']
        if risk['high_volatility_symbols'] > len(results['symbols']) * 0.3:
            recommendations.append("⚠ High portfolio volatility, consider risk management")
        
        if risk['concentrated_positions'] > 0:
            recommendations.append("⚠ Position concentration risk detected")
        
        # Sector recommendations
        sectors = summary['sectors']
        tech_weight = sectors.get('Technology', {}).get('total_weight', 0)
        if tech_weight > 0.6:
            recommendations.append("⚠ High technology sector concentration")
        
        # Trading recommendations
        opportunities = summary['top_opportunities']
        if opportunities:
            best_opp = opportunities[0]
            recommendations.append(
                f"→ Best opportunity: {best_opp['symbol']} "
                f"({best_opp['signal_type']}, conf: {best_opp['confidence']:.2f})"
            )
        
        for rec in recommendations:
            print(f"  {rec}")


def main():
    """Main portfolio analysis example."""
    print("=== Elliott Wave Portfolio Analysis ===")
    
    # Define sample portfolios
    portfolios = {
        'tech_growth': {
            'AAPL': 0.25,
            'MSFT': 0.20,
            'GOOGL': 0.20,
            'NVDA': 0.15,
            'TSLA': 0.10,
            'META': 0.10
        },
        'diversified': {
            'AAPL': 0.15,
            'MSFT': 0.15,
            'JPM': 0.10,
            'JNJ': 0.10,
            'TSLA': 0.10,
            'GOOGL': 0.10,
            'UNH': 0.10,
            'NVDA': 0.10,
            'BAC': 0.05,
            'PFE': 0.05
        },
        'sp500_top': {
            'AAPL': 0.07,
            'MSFT': 0.06,
            'AMZN': 0.04,
            'NVDA': 0.04,
            'GOOGL': 0.04,
            'TSLA': 0.03,
            'META': 0.03,
            'BRK-B': 0.02,
            'UNH': 0.02,
            'JNJ': 0.02,
            'JPM': 0.02,
            'V': 0.02,
            'PG': 0.02,
            'HD': 0.02,
            'CVX': 0.02,
            'MA': 0.02,
            'PFE': 0.02,
            'ABBV': 0.02,
            'BAC': 0.02,
            'KO': 0.02,
            'PEP': 0.02,
            'AVGO': 0.02,
            'TMO': 0.02,
            'COST': 0.02,
            'DIS': 0.02,
            'ABT': 0.02,
            'ACN': 0.02,
            'LLY': 0.02,
            'WMT': 0.02,
            'VZ': 0.02,
            'ADBE': 0.02,
            'CMCSA': 0.02,
            'MRK': 0.02,
            'NFLX': 0.02,
            'CRM': 0.02,
            'DHR': 0.02,
            'NKE': 0.02,
            'TXN': 0.02,
            'NEE': 0.02,
            'ORCL': 0.02,
            'QCOM': 0.02,
            'PM': 0.02,
            'MDT': 0.02,
            'UPS': 0.02,
            'T': 0.02,
            'HON': 0.02,
            'LOW': 0.02
        }
    }
    
    analyzer = PortfolioAnalyzer()
    
    # Analyze each portfolio
    for portfolio_name, portfolio in portfolios.items():
        print(f"\n{'='*100}")
        print(f"ANALYZING {portfolio_name.upper().replace('_', ' ')} PORTFOLIO")
        print(f"{'='*100}")
        
        try:
            # Limit to top 10 for demo purposes
            if len(portfolio) > 10:
                sorted_portfolio = dict(sorted(portfolio.items(), key=lambda x: x[1], reverse=True)[:10])
                # Normalize weights
                total_weight = sum(sorted_portfolio.values())
                sorted_portfolio = {k: v/total_weight for k, v in sorted_portfolio.items()}
            else:
                sorted_portfolio = portfolio
            
            results = analyzer.analyze_portfolio(sorted_portfolio, period="6mo")
            analyzer.generate_portfolio_report(results)
            
        except Exception as e:
            print(f"Error analyzing {portfolio_name}: {e}")
            continue


def compare_portfolios():
    """Compare different portfolio strategies."""
    print(f"\n{'='*60}")
    print("PORTFOLIO COMPARISON")
    print(f"{'='*60}")
    
    portfolios = {
        'Growth': {
            'TSLA': 0.30,
            'NVDA': 0.25,
            'NFLX': 0.20,
            'AMZN': 0.25
        },
        'Value': {
            'JPM': 0.25,
            'JNJ': 0.25,
            'PG': 0.25,
            'KO': 0.25
        },
        'Balanced': {
            'AAPL': 0.25,
            'JPM': 0.25,
            'JNJ': 0.25,
            'TSLA': 0.25
        }
    }
    
    analyzer = PortfolioAnalyzer()
    comparison_results = {}
    
    for name, portfolio in portfolios.items():
        try:
            results = analyzer.analyze_portfolio(portfolio, period="3mo")
            comparison_results[name] = results['summary']
            
            print(f"\n{name} Portfolio:")
            print(f"  Portfolio Score: {results['summary']['portfolio_score']:.1f}")
            print(f"  High Conf Signals: {results['summary']['high_conf_signals']}")
            print(f"  Top Opportunity: {results['summary']['top_opportunities'][0]['symbol'] if results['summary']['top_opportunities'] else 'None'}")
            
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
    
    # Summary comparison
    if comparison_results:
        print(f"\n{'Portfolio':<12} {'Score':<8} {'Signals':<8} {'Risk':<8}")
        print("-" * 40)
        
        for name, summary in comparison_results.items():
            risk_level = "High" if summary['risk_assessment']['high_volatility_symbols'] > 0 else "Low"
            print(f"{name:<12} {summary['portfolio_score']:<7.1f} "
                  f"{summary['high_conf_signals']:<8} {risk_level:<8}")


if __name__ == "__main__":
    main()
    compare_portfolios()
    
    print(f"\n{'='*60}")
    print("Portfolio analysis completed!")
    print("Use this framework to analyze your own portfolios.")
    print(f"{'='*60}")
