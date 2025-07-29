"""
Comprehensive backtesting engine for Elliott Wave trading strategies.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum

from ..utils.logger import get_logger, TradingLogger
from ..utils.config import get_config
from ..trading.strategy import ElliottWaveStrategy, TradingSignal, SignalType, Position, PositionType

logger = get_logger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Represents a trading order."""
    id: str
    timestamp: pd.Timestamp
    symbol: str
    order_type: OrderType
    side: str  # BUY or SELL
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_time: Optional[pd.Timestamp] = None
    commission: float = 0.0


@dataclass
class Trade:
    """Represents a completed trade."""
    id: str
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # LONG or SHORT
    pnl: float
    pnl_pct: float
    commission: float
    signal_confidence: float = 0.0
    wave_type: str = ""
    duration_hours: float = 0.0
    
    def __post_init__(self):
        if isinstance(self.entry_time, str):
            self.entry_time = pd.Timestamp(self.entry_time)
        if isinstance(self.exit_time, str):
            self.exit_time = pd.Timestamp(self.exit_time)
        
        if self.duration_hours == 0.0:
            self.duration_hours = (self.exit_time - self.entry_time).total_seconds() / 3600


@dataclass
class BacktestResults:
    """Comprehensive backtesting results."""
    
    # Basic metrics
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    volatility: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Time-based metrics
    avg_holding_period_hours: float = 0.0
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    
    # Portfolio metrics
    initial_capital: float = 0.0
    final_capital: float = 0.0
    peak_capital: float = 0.0
    
    # Additional data
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_curve: pd.Series = field(default_factory=pd.Series)


class BacktestEngine:
    """
    Comprehensive backtesting engine for Elliott Wave strategies.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize backtesting engine.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.trading_logger = TradingLogger()
        
        # Backtesting parameters
        self.initial_capital = self.config.get('backtesting.initial_capital', 100000)
        self.commission_rate = self.config.get('backtesting.commission', 0.001)
        self.slippage_rate = self.config.get('backtesting.slippage', 0.0005)
        self.max_positions = self.config.get('backtesting.max_positions', 5)
        
        # Portfolio state
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        
        # Order management
        self._order_id_counter = 0
        self._trade_id_counter = 0
        
        logger.info(f"BacktestEngine initialized with ${self.initial_capital:,.2f} initial capital")
    
    def run_backtest(
        self,
        strategy: ElliottWaveStrategy,
        data: pd.DataFrame,
        symbol: str = "SYMBOL",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> BacktestResults:
        """
        Run complete backtest.
        
        Args:
            strategy: Trading strategy to test
            data: OHLCV DataFrame
            symbol: Trading symbol
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            BacktestResults object
        """
        try:
            logger.info(f"Starting backtest for {symbol}")
            
            # Filter data by date range
            test_data = self._filter_data_by_date(data, start_date, end_date)
            
            if len(test_data) < 50:
                raise ValueError("Insufficient data for backtesting")
            
            # Reset portfolio state
            self._reset_portfolio()
            
            # Generate signals
            signals = strategy.generate_signals(test_data, symbol)
            logger.info(f"Generated {len(signals)} signals for backtesting")
            
            # Execute trades based on signals
            self._execute_signals(signals, test_data, symbol)
            
            # Calculate results
            results = self._calculate_results(test_data, symbol)
            
            # Log summary
            self._log_backtest_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise
    
    def _filter_data_by_date(
        self,
        data: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """
        Filter data by date range.
        
        Args:
            data: OHLCV DataFrame
            start_date: Start date string
            end_date: End date string
            
        Returns:
            Filtered DataFrame
        """
        filtered_data = data.copy()
        
        if start_date:
            filtered_data = filtered_data[filtered_data.index >= pd.Timestamp(start_date)]
        
        if end_date:
            filtered_data = filtered_data[filtered_data.index <= pd.Timestamp(end_date)]
        
        return filtered_data
    
    def _reset_portfolio(self):
        """Reset portfolio to initial state."""
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self._order_id_counter = 0
        self._trade_id_counter = 0
    
    def _execute_signals(self, signals: List[TradingSignal], data: pd.DataFrame, symbol: str):
        """
        Execute trading signals and simulate orders.
        
        Args:
            signals: List of trading signals
            data: OHLCV DataFrame
            symbol: Trading symbol
        """
        for signal in signals:
            try:
                # Find the closest data point to signal timestamp
                signal_data = self._get_signal_data(signal, data)
                if signal_data is None:
                    continue
                
                # Execute signal
                if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                    self._execute_entry_signal(signal, signal_data, symbol)
                elif signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
                    self._execute_exit_signal(signal, signal_data, symbol)
                
                # Update equity curve
                self._update_equity_curve(signal.timestamp, symbol, data)
                
            except Exception as e:
                logger.debug(f"Error executing signal: {e}")
                continue
    
    def _get_signal_data(self, signal: TradingSignal, data: pd.DataFrame) -> Optional[pd.Series]:
        """
        Get market data at signal timestamp.
        
        Args:
            signal: Trading signal
            data: OHLCV DataFrame
            
        Returns:
            Market data or None
        """
        try:
            # Find closest timestamp
            idx = data.index.get_indexer([signal.timestamp], method='nearest')[0]
            if idx == -1:
                return None
            
            return data.iloc[idx]
            
        except Exception as e:
            logger.debug(f"Error getting signal data: {e}")
            return None
    
    def _execute_entry_signal(self, signal: TradingSignal, market_data: pd.Series, symbol: str):
        """
        Execute entry signal (open position).
        
        Args:
            signal: Trading signal
            market_data: Market data at signal time
            symbol: Trading symbol
        """
        try:
            # Check if we already have a position in this symbol
            if symbol in self.positions and self.positions[symbol].position_type != PositionType.FLAT:
                logger.debug(f"Already have position in {symbol}, skipping signal")
                return
            
            # Check position limit
            active_positions = sum(1 for pos in self.positions.values() if pos.position_type != PositionType.FLAT)
            if active_positions >= self.max_positions:
                logger.debug(f"Maximum positions ({self.max_positions}) reached, skipping signal")
                return
            
            # Calculate position size
            position_size = self._calculate_position_size(signal, market_data)
            if position_size <= 0:
                return
            
            # Determine execution price (including slippage)
            if signal.signal_type == SignalType.BUY:
                execution_price = market_data['high'] * (1 + self.slippage_rate)  # Slippage against us
                position_type = PositionType.LONG
            else:  # SELL
                execution_price = market_data['low'] * (1 - self.slippage_rate)  # Slippage against us
                position_type = PositionType.SHORT
            
            # Calculate commission
            commission = execution_price * position_size * self.commission_rate
            
            # Check if we have enough capital
            required_capital = execution_price * position_size + commission
            if required_capital > self.current_capital:
                logger.debug(f"Insufficient capital for trade: required ${required_capital:.2f}, available ${self.current_capital:.2f}")
                return
            
            # Create and execute order
            order = Order(
                id=f"ORDER_{self._order_id_counter}",
                timestamp=signal.timestamp,
                symbol=symbol,
                order_type=OrderType.MARKET,
                side=signal.signal_type.value,
                quantity=position_size,
                price=execution_price,
                status=OrderStatus.FILLED,
                fill_price=execution_price,
                fill_time=signal.timestamp,
                commission=commission
            )
            
            self.orders.append(order)
            self._order_id_counter += 1
            
            # Create position
            position = Position(
                symbol=symbol,
                position_type=position_type,
                entry_price=execution_price,
                entry_time=signal.timestamp,
                quantity=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                current_price=execution_price
            )
            
            self.positions[symbol] = position
            
            # Update capital
            self.current_capital -= required_capital
            
            # Log the trade
            self.trading_logger.log_trade(
                symbol=symbol,
                action=signal.signal_type.value,
                quantity=position_size,
                price=execution_price,
                timestamp=signal.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                strategy="Elliott Wave",
                confidence=signal.confidence
            )
            
            logger.debug(f"Opened {position_type.value} position in {symbol} at ${execution_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error executing entry signal: {e}")
    
    def _execute_exit_signal(self, signal: TradingSignal, market_data: pd.Series, symbol: str):
        """
        Execute exit signal (close position).
        
        Args:
            signal: Trading signal
            market_data: Market data at signal time
            symbol: Trading symbol
        """
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            
            if position.position_type == PositionType.FLAT:
                return
            
            # Determine execution price
            if position.position_type == PositionType.LONG:
                execution_price = market_data['low'] * (1 - self.slippage_rate)  # Slippage against us
            else:  # SHORT
                execution_price = market_data['high'] * (1 + self.slippage_rate)  # Slippage against us
            
            # Close position
            self._close_position(symbol, execution_price, signal.timestamp, signal.confidence, signal.wave_type or "")
            
        except Exception as e:
            logger.error(f"Error executing exit signal: {e}")
    
    def _calculate_position_size(self, signal: TradingSignal, market_data: pd.Series) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            signal: Trading signal
            market_data: Market data
            
        Returns:
            Position size
        """
        try:
            # Risk per trade (percentage of capital)
            risk_per_trade = self.config.get('backtesting.risk_per_trade', 0.02)
            risk_amount = self.current_capital * risk_per_trade
            
            # Calculate position size based on stop loss
            if signal.stop_loss:
                price = signal.price
                stop_distance = abs(price - signal.stop_loss)
                
                if stop_distance > 0:
                    position_size = risk_amount / stop_distance
                else:
                    position_size = 0
            else:
                # Default position size if no stop loss
                position_size = risk_amount / (signal.price * 0.02)  # Assume 2% risk
            
            # Ensure we don't exceed available capital
            max_position_value = self.current_capital * 0.95  # Use max 95% of capital
            max_position_size = max_position_value / signal.price
            
            position_size = min(position_size, max_position_size)
            
            return max(0, position_size)
            
        except Exception as e:
            logger.debug(f"Error calculating position size: {e}")
            return 0
    
    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_time: pd.Timestamp,
        signal_confidence: float = 0.0,
        wave_type: str = ""
    ):
        """
        Close a position and record the trade.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            exit_time: Exit timestamp
            signal_confidence: Signal confidence
            wave_type: Wave type
        """
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            
            if position.position_type == PositionType.FLAT:
                return
            
            # Calculate commission
            commission = exit_price * position.quantity * self.commission_rate
            
            # Calculate P&L
            if position.position_type == PositionType.LONG:
                gross_pnl = (exit_price - position.entry_price) * position.quantity
                side = "LONG"
            else:  # SHORT
                gross_pnl = (position.entry_price - exit_price) * position.quantity
                side = "SHORT"
            
            net_pnl = gross_pnl - commission
            pnl_pct = net_pnl / (position.entry_price * position.quantity)
            
            # Update capital
            proceeds = exit_price * position.quantity - commission
            self.current_capital += proceeds
            
            # Update peak capital
            self.peak_capital = max(self.peak_capital, self.current_capital)
            
            # Create trade record
            trade = Trade(
                id=f"TRADE_{self._trade_id_counter}",
                symbol=symbol,
                entry_time=position.entry_time,
                exit_time=exit_time,
                entry_price=position.entry_price,
                exit_price=exit_price,
                quantity=position.quantity,
                side=side,
                pnl=net_pnl,
                pnl_pct=pnl_pct,
                commission=commission,
                signal_confidence=signal_confidence,
                wave_type=wave_type
            )
            
            self.trades.append(trade)
            self._trade_id_counter += 1
            
            # Remove position
            position.position_type = PositionType.FLAT
            del self.positions[symbol]
            
            # Log the trade
            self.trading_logger.log_trade(
                symbol=symbol,
                action="CLOSE_" + side,
                quantity=position.quantity,
                price=exit_price,
                timestamp=exit_time.strftime('%Y-%m-%d %H:%M:%S'),
                strategy="Elliott Wave",
                confidence=signal_confidence
            )
            
            logger.debug(f"Closed {side} position in {symbol} at ${exit_price:.2f}, P&L: ${net_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def _update_equity_curve(self, timestamp: pd.Timestamp, symbol: str, data: pd.DataFrame):
        """
        Update equity curve with current portfolio value.
        
        Args:
            timestamp: Current timestamp
            symbol: Trading symbol
            data: Market data
        """
        try:
            # Calculate current portfolio value
            portfolio_value = self.current_capital
            
            # Add unrealized P&L from open positions
            for pos_symbol, position in self.positions.items():
                if position.position_type != PositionType.FLAT:
                    # Get current market price
                    try:
                        current_data = data.loc[timestamp]
                        current_price = current_data['close']
                        
                        # Calculate unrealized P&L
                        if position.position_type == PositionType.LONG:
                            unrealized_pnl = (current_price - position.entry_price) * position.quantity
                        else:  # SHORT
                            unrealized_pnl = (position.entry_price - current_price) * position.quantity
                        
                        portfolio_value += unrealized_pnl
                        
                    except KeyError:
                        # If timestamp not in data, skip
                        pass
            
            self.equity_curve.append((timestamp, portfolio_value))
            
        except Exception as e:
            logger.debug(f"Error updating equity curve: {e}")
    
    def _calculate_results(self, data: pd.DataFrame, symbol: str) -> BacktestResults:
        """
        Calculate comprehensive backtest results.
        
        Args:
            data: Market data
            symbol: Trading symbol
            
        Returns:
            BacktestResults object
        """
        try:
            results = BacktestResults()
            
            # Basic information
            results.initial_capital = self.initial_capital
            results.final_capital = self.current_capital
            results.peak_capital = self.peak_capital
            results.start_date = data.index[0]
            results.end_date = data.index[-1]
            results.trades = self.trades.copy()
            
            # Calculate total return
            results.total_return = self.current_capital - self.initial_capital
            results.total_return_pct = results.total_return / self.initial_capital
            
            # Calculate annualized return
            days = (results.end_date - results.start_date).days
            years = days / 365.25
            if years > 0:
                results.annual_return_pct = ((results.final_capital / results.initial_capital) ** (1/years)) - 1
            
            # Trade statistics
            if self.trades:
                results.total_trades = len(self.trades)
                results.winning_trades = sum(1 for trade in self.trades if trade.pnl > 0)
                results.losing_trades = sum(1 for trade in self.trades if trade.pnl < 0)
                results.win_rate = results.winning_trades / results.total_trades
                
                winning_pnls = [trade.pnl for trade in self.trades if trade.pnl > 0]
                losing_pnls = [trade.pnl for trade in self.trades if trade.pnl < 0]
                
                results.avg_win = np.mean(winning_pnls) if winning_pnls else 0
                results.avg_loss = np.mean(losing_pnls) if losing_pnls else 0
                results.largest_win = max(winning_pnls) if winning_pnls else 0
                results.largest_loss = min(losing_pnls) if losing_pnls else 0
                
                gross_profit = sum(winning_pnls) if winning_pnls else 0
                gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0
                results.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                # Average holding period
                durations = [trade.duration_hours for trade in self.trades]
                results.avg_holding_period_hours = np.mean(durations) if durations else 0
            
            # Create equity curve series
            if self.equity_curve:
                equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
                equity_df.set_index('timestamp', inplace=True)
                results.equity_curve = equity_df['equity']
                
                # Calculate drawdown
                results.drawdown_curve = self._calculate_drawdown(results.equity_curve)
                results.max_drawdown = results.drawdown_curve.min()
                results.max_drawdown_pct = results.max_drawdown / results.peak_capital
                
                # Calculate volatility and Sharpe ratio
                returns = results.equity_curve.pct_change().dropna()
                if len(returns) > 1:
                    results.volatility = returns.std() * np.sqrt(252)  # Annualized
                    
                    if results.volatility > 0:
                        risk_free_rate = 0.02  # Assume 2% risk-free rate
                        excess_return = results.annual_return_pct - risk_free_rate
                        results.sharpe_ratio = excess_return / results.volatility
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating results: {e}")
            return BacktestResults()
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """
        Calculate drawdown series.
        
        Args:
            equity_curve: Equity curve series
            
        Returns:
            Drawdown series
        """
        try:
            peak = equity_curve.expanding().max()
            drawdown = equity_curve - peak
            return drawdown
            
        except Exception as e:
            logger.debug(f"Error calculating drawdown: {e}")
            return pd.Series(dtype=float)
    
    def _log_backtest_summary(self, results: BacktestResults):
        """
        Log backtest summary.
        
        Args:
            results: Backtest results
        """
        self.trading_logger.log_backtest_result(
            strategy="Elliott Wave",
            total_return=results.total_return_pct * 100,
            sharpe_ratio=results.sharpe_ratio,
            max_drawdown=results.max_drawdown_pct * 100,
            win_rate=results.win_rate * 100,
            total_trades=results.total_trades
        )
    
    def generate_report(self, results: BacktestResults) -> Dict[str, Any]:
        """
        Generate detailed backtest report.
        
        Args:
            results: Backtest results
            
        Returns:
            Report dictionary
        """
        report = {
            'summary': {
                'initial_capital': results.initial_capital,
                'final_capital': results.final_capital,
                'total_return': results.total_return,
                'total_return_pct': results.total_return_pct * 100,
                'annual_return_pct': results.annual_return_pct * 100,
                'max_drawdown_pct': results.max_drawdown_pct * 100,
                'sharpe_ratio': results.sharpe_ratio,
                'volatility': results.volatility,
                'profit_factor': results.profit_factor
            },
            'trade_stats': {
                'total_trades': results.total_trades,
                'winning_trades': results.winning_trades,
                'losing_trades': results.losing_trades,
                'win_rate': results.win_rate * 100,
                'avg_win': results.avg_win,
                'avg_loss': results.avg_loss,
                'largest_win': results.largest_win,
                'largest_loss': results.largest_loss,
                'avg_holding_period_hours': results.avg_holding_period_hours
            },
            'period': {
                'start_date': results.start_date.strftime('%Y-%m-%d') if results.start_date else None,
                'end_date': results.end_date.strftime('%Y-%m-%d') if results.end_date else None,
                'duration_days': (results.end_date - results.start_date).days if results.start_date and results.end_date else 0
            }
        }
        
        return report


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.data.data_loader import DataLoader
    from src.trading.strategy import ElliottWaveStrategy
    
    # Load sample data
    loader = DataLoader()
    data = loader.get_yahoo_data("AAPL", period="2y")
    
    # Initialize strategy and backtester
    strategy = ElliottWaveStrategy()
    backtester = BacktestEngine()
    
    # Run backtest
    results = backtester.run_backtest(strategy, data, "AAPL")
    
    # Generate report
    report = backtester.generate_report(results)
    
    print("=== BACKTEST RESULTS ===")
    print(f"Initial Capital: ${report['summary']['initial_capital']:,.2f}")
    print(f"Final Capital: ${report['summary']['final_capital']:,.2f}")
    print(f"Total Return: {report['summary']['total_return_pct']:.2f}%")
    print(f"Annual Return: {report['summary']['annual_return_pct']:.2f}%")
    print(f"Max Drawdown: {report['summary']['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {report['summary']['sharpe_ratio']:.2f}")
    print(f"Win Rate: {report['trade_stats']['win_rate']:.2f}%")
    print(f"Total Trades: {report['trade_stats']['total_trades']}")
    print(f"Profit Factor: {report['summary']['profit_factor']:.2f}")
    
    if results.trades:
        print(f"\nSample Trades:")
        for i, trade in enumerate(results.trades[:5]):
            print(f"  Trade {i+1}: {trade.side} {trade.symbol} "
                  f"Entry: ${trade.entry_price:.2f} "
                  f"Exit: ${trade.exit_price:.2f} "
                  f"P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2%})")
    
    print(f"\nBacktest Period: {report['period']['start_date']} to {report['period']['end_date']}")
    print(f"Duration: {report['period']['duration_days']} days")
