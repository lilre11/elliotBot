"""
Elliott Wave trading strategy implementation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import logging
from datetime import datetime

from ..utils.logger import get_logger, TradingLogger
from ..utils.config import get_config
from ..analysis.wave_detector import Wave, WaveType, TrendDirection, WaveDetector
from ..analysis.fibonacci import FibonacciAnalyzer, FibonacciAnalysis

logger = get_logger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


class PositionType(Enum):
    """Position types."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class TradingSignal:
    """Represents a trading signal."""
    timestamp: pd.Timestamp
    symbol: str
    signal_type: SignalType
    price: float
    confidence: float
    wave_type: Optional[str] = None
    fibonacci_level: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = pd.Timestamp(self.timestamp)


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    position_type: PositionType
    entry_price: float
    entry_time: pd.Timestamp
    quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: Optional[float] = None
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.current_price is None:
            return 0.0
        
        if self.position_type == PositionType.LONG:
            return (self.current_price - self.entry_price) * self.quantity
        elif self.position_type == PositionType.SHORT:
            return (self.entry_price - self.current_price) * self.quantity
        else:
            return 0.0
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.position_type == PositionType.FLAT:
            return 0.0
        return self.unrealized_pnl / (self.entry_price * self.quantity)


class ElliottWaveStrategy:
    """
    Elliott Wave trading strategy implementation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Elliott Wave strategy.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.wave_detector = WaveDetector(config_path)
        self.fibonacci_analyzer = FibonacciAnalyzer(config_path)
        self.trading_logger = TradingLogger()
        
        # Strategy parameters
        self.min_confidence = self.config.get('wave_detection.confidence_threshold', 0.7)
        self.risk_per_trade = self.config.get('backtesting.risk_per_trade', 0.02)
        self.max_positions = self.config.get('backtesting.max_positions', 5)
        
        # Current positions
        self.positions: Dict[str, Position] = {}
        
        logger.info("ElliottWaveStrategy initialized")
    
    def generate_signals(self, data: pd.DataFrame, symbol: str = "SYMBOL") -> List[TradingSignal]:
        """
        Generate trading signals based on Elliott Wave analysis.
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            
        Returns:
            List of trading signals
        """
        try:
            signals = []
            
            # Detect Elliott Waves
            waves = self.wave_detector.detect_waves(data)
            
            if not waves:
                logger.debug("No waves detected, no signals generated")
                return signals
            
            # Get Fibonacci analysis
            fibonacci_analysis = self._get_fibonacci_analysis(data)
            
            # Analyze each wave for trading opportunities
            for wave in waves:
                wave_signals = self._analyze_wave_for_signals(wave, data, symbol, fibonacci_analysis)
                signals.extend(wave_signals)
            
            # Filter and validate signals
            validated_signals = self._validate_signals(signals, data)
            
            logger.info(f"Generated {len(validated_signals)} trading signals for {symbol}")
            return validated_signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    def _get_fibonacci_analysis(self, data: pd.DataFrame) -> Optional[FibonacciAnalysis]:
        """
        Get Fibonacci analysis for the data.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            FibonacciAnalysis or None
        """
        try:
            if len(data) < 50:
                return None
            
            # Find recent swing high and low
            lookback = min(50, len(data))
            recent_data = data.tail(lookback)
            
            swing_high = recent_data['high'].max()
            swing_low = recent_data['low'].min()
            current_price = data['close'].iloc[-1]
            
            # Determine trend direction
            trend_direction = 'up' if current_price > data['close'].rolling(20).mean().iloc[-1] else 'down'
            
            return self.fibonacci_analyzer.analyze_retracement(
                swing_high, swing_low, current_price, trend_direction
            )
            
        except Exception as e:
            logger.debug(f"Error getting Fibonacci analysis: {e}")
            return None
    
    def _analyze_wave_for_signals(
        self, 
        wave: Wave, 
        data: pd.DataFrame, 
        symbol: str,
        fibonacci_analysis: Optional[FibonacciAnalysis]
    ) -> List[TradingSignal]:
        """
        Analyze a wave for trading signals.
        
        Args:
            wave: Wave to analyze
            data: OHLCV DataFrame
            symbol: Trading symbol
            fibonacci_analysis: Fibonacci analysis
            
        Returns:
            List of trading signals
        """
        signals = []
        
        try:
            # Only generate signals for high-confidence waves
            if wave.confidence < self.min_confidence:
                return signals
            
            current_price = data['close'].iloc[-1]
            current_time = data.index[-1]
            
            # Strategy 1: End of corrective waves (Wave 2, 4, B)
            if wave.wave_type in [WaveType.IMPULSE_2, WaveType.IMPULSE_4, WaveType.CORRECTIVE_B]:
                signal = self._generate_corrective_end_signal(wave, current_price, current_time, symbol, fibonacci_analysis)
                if signal:
                    signals.append(signal)
            
            # Strategy 2: Wave 3 momentum (strongest impulse wave)
            elif wave.wave_type == WaveType.IMPULSE_3:
                signal = self._generate_wave3_signal(wave, current_price, current_time, symbol, fibonacci_analysis)
                if signal:
                    signals.append(signal)
            
            # Strategy 3: Wave 5 completion
            elif wave.wave_type == WaveType.IMPULSE_5:
                signal = self._generate_wave5_completion_signal(wave, current_price, current_time, symbol, fibonacci_analysis)
                if signal:
                    signals.append(signal)
            
            # Strategy 4: Corrective wave completion (Wave C)
            elif wave.wave_type == WaveType.CORRECTIVE_C:
                signal = self._generate_corrective_completion_signal(wave, current_price, current_time, symbol, fibonacci_analysis)
                if signal:
                    signals.append(signal)
            
        except Exception as e:
            logger.debug(f"Error analyzing wave for signals: {e}")
        
        return signals
    
    def _generate_corrective_end_signal(
        self, 
        wave: Wave, 
        current_price: float, 
        current_time: pd.Timestamp, 
        symbol: str,
        fibonacci_analysis: Optional[FibonacciAnalysis]
    ) -> Optional[TradingSignal]:
        """
        Generate signal at the end of corrective waves.
        
        Args:
            wave: Corrective wave
            current_price: Current price
            current_time: Current timestamp
            symbol: Trading symbol
            fibonacci_analysis: Fibonacci analysis
            
        Returns:
            Trading signal or None
        """
        # Check if we're near the end of the corrective wave
        if abs(current_time - wave.end_point.timestamp).days > 5:
            return None
        
        # Determine signal direction based on expected next wave
        if wave.wave_type in [WaveType.IMPULSE_2, WaveType.IMPULSE_4]:
            # After Wave 2/4, expect impulsive move in original trend direction
            signal_type = SignalType.BUY if wave.direction == TrendDirection.DOWN else SignalType.SELL
            next_wave = "Wave 3" if wave.wave_type == WaveType.IMPULSE_2 else "Wave 5"
        else:  # Wave B
            # After Wave B, expect Wave C in direction of Wave A
            signal_type = SignalType.BUY if wave.direction == TrendDirection.UP else SignalType.SELL
            next_wave = "Wave C"
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_stop_take_profit(wave, signal_type, fibonacci_analysis)
        
        # Check Fibonacci confluence
        fib_level = None
        if fibonacci_analysis:
            price_analysis = self.fibonacci_analyzer.analyze_price_at_fibonacci(current_price, fibonacci_analysis)
            if price_analysis['at_fibonacci_level']:
                fib_level = price_analysis['nearest_levels'][0]['level'].ratio
        
        return TradingSignal(
            timestamp=current_time,
            symbol=symbol,
            signal_type=signal_type,
            price=current_price,
            confidence=wave.confidence,
            wave_type=wave.wave_type.value,
            fibonacci_level=fib_level,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=f"End of {wave.wave_type.value}, expecting {next_wave}"
        )
    
    def _generate_wave3_signal(
        self, 
        wave: Wave, 
        current_price: float, 
        current_time: pd.Timestamp, 
        symbol: str,
        fibonacci_analysis: Optional[FibonacciAnalysis]
    ) -> Optional[TradingSignal]:
        """
        Generate signal during Wave 3 momentum.
        
        Args:
            wave: Wave 3
            current_price: Current price
            current_time: Current timestamp
            symbol: Trading symbol
            fibonacci_analysis: Fibonacci analysis
            
        Returns:
            Trading signal or None
        """
        # Only signal if we're in the early part of Wave 3
        wave_progress = (current_time - wave.start_point.timestamp) / (wave.end_point.timestamp - wave.start_point.timestamp)
        
        if wave_progress > 0.5:  # Too late in the wave
            return None
        
        # Wave 3 should be in the direction of the trend
        signal_type = SignalType.BUY if wave.direction == TrendDirection.UP else SignalType.SELL
        
        # Calculate aggressive stop loss and take profit for momentum trade
        stop_loss, take_profit = self._calculate_stop_take_profit(wave, signal_type, fibonacci_analysis, aggressive=True)
        
        return TradingSignal(
            timestamp=current_time,
            symbol=symbol,
            signal_type=signal_type,
            price=current_price,
            confidence=min(wave.confidence * 1.2, 1.0),  # Boost confidence for Wave 3
            wave_type=wave.wave_type.value,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason="Wave 3 momentum trade - strongest impulse wave"
        )
    
    def _generate_wave5_completion_signal(
        self, 
        wave: Wave, 
        current_price: float, 
        current_time: pd.Timestamp, 
        symbol: str,
        fibonacci_analysis: Optional[FibonacciAnalysis]
    ) -> Optional[TradingSignal]:
        """
        Generate signal for Wave 5 completion (reversal).
        
        Args:
            wave: Wave 5
            current_price: Current price
            current_time: Current timestamp
            symbol: Trading symbol
            fibonacci_analysis: Fibonacci analysis
            
        Returns:
            Trading signal or None
        """
        # Check if we're near the completion of Wave 5
        if abs(current_time - wave.end_point.timestamp).days > 3:
            return None
        
        # Wave 5 completion suggests reversal
        signal_type = SignalType.SELL if wave.direction == TrendDirection.UP else SignalType.BUY
        
        # More conservative stop/take profit for reversal trades
        stop_loss, take_profit = self._calculate_stop_take_profit(wave, signal_type, fibonacci_analysis, conservative=True)
        
        return TradingSignal(
            timestamp=current_time,
            symbol=symbol,
            signal_type=signal_type,
            price=current_price,
            confidence=wave.confidence * 0.8,  # Lower confidence for reversal
            wave_type=wave.wave_type.value,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason="Wave 5 completion - expecting corrective phase"
        )
    
    def _generate_corrective_completion_signal(
        self, 
        wave: Wave, 
        current_price: float, 
        current_time: pd.Timestamp, 
        symbol: str,
        fibonacci_analysis: Optional[FibonacciAnalysis]
    ) -> Optional[TradingSignal]:
        """
        Generate signal for corrective wave completion.
        
        Args:
            wave: Wave C
            current_price: Current price
            current_time: Current timestamp
            symbol: Trading symbol
            fibonacci_analysis: Fibonacci analysis
            
        Returns:
            Trading signal or None
        """
        # Check if we're near the completion of Wave C
        if abs(current_time - wave.end_point.timestamp).days > 3:
            return None
        
        # Wave C completion suggests new impulse cycle
        signal_type = SignalType.BUY if wave.direction == TrendDirection.DOWN else SignalType.SELL
        
        stop_loss, take_profit = self._calculate_stop_take_profit(wave, signal_type, fibonacci_analysis)
        
        return TradingSignal(
            timestamp=current_time,
            symbol=symbol,
            signal_type=signal_type,
            price=current_price,
            confidence=wave.confidence,
            wave_type=wave.wave_type.value,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason="Wave C completion - expecting new impulse cycle"
        )
    
    def _calculate_stop_take_profit(
        self, 
        wave: Wave, 
        signal_type: SignalType, 
        fibonacci_analysis: Optional[FibonacciAnalysis],
        aggressive: bool = False,
        conservative: bool = False
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            wave: Current wave
            signal_type: Signal type
            fibonacci_analysis: Fibonacci analysis
            aggressive: Use aggressive targets
            conservative: Use conservative targets
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        try:
            wave_height = abs(wave.end_point.price - wave.start_point.price)
            current_price = wave.end_point.price
            
            # Base risk/reward ratios
            if aggressive:
                stop_ratio = 0.5  # Tighter stop
                profit_ratio = 2.0  # Higher target
            elif conservative:
                stop_ratio = 1.0  # Wider stop
                profit_ratio = 1.5  # Lower target
            else:
                stop_ratio = 0.75  # Standard stop
                profit_ratio = 2.0  # Standard target
            
            # Calculate levels based on signal direction
            if signal_type in [SignalType.BUY]:
                stop_loss = current_price - wave_height * stop_ratio
                take_profit = current_price + wave_height * profit_ratio
            elif signal_type in [SignalType.SELL]:
                stop_loss = current_price + wave_height * stop_ratio
                take_profit = current_price - wave_height * profit_ratio
            else:
                return None, None
            
            # Adjust using Fibonacci levels if available
            if fibonacci_analysis:
                stop_loss, take_profit = self._adjust_levels_with_fibonacci(
                    stop_loss, take_profit, signal_type, fibonacci_analysis
                )
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.debug(f"Error calculating stop/take profit: {e}")
            return None, None
    
    def _adjust_levels_with_fibonacci(
        self, 
        stop_loss: float, 
        take_profit: float, 
        signal_type: SignalType,
        fibonacci_analysis: FibonacciAnalysis
    ) -> Tuple[float, float]:
        """
        Adjust stop loss and take profit using Fibonacci levels.
        
        Args:
            stop_loss: Initial stop loss
            take_profit: Initial take profit
            signal_type: Signal type
            fibonacci_analysis: Fibonacci analysis
            
        Returns:
            Adjusted (stop_loss, take_profit)
        """
        try:
            # Find nearest Fibonacci levels
            fib_levels = [level.price for level in fibonacci_analysis.key_levels]
            
            if signal_type == SignalType.BUY:
                # Adjust stop loss to nearest support below
                support_levels = [level for level in fib_levels if level < stop_loss]
                if support_levels:
                    stop_loss = max(support_levels)
                
                # Adjust take profit to nearest resistance above
                resistance_levels = [level for level in fib_levels if level > take_profit]
                if resistance_levels:
                    take_profit = min(resistance_levels)
            
            elif signal_type == SignalType.SELL:
                # Adjust stop loss to nearest resistance above
                resistance_levels = [level for level in fib_levels if level > stop_loss]
                if resistance_levels:
                    stop_loss = min(resistance_levels)
                
                # Adjust take profit to nearest support below
                support_levels = [level for level in fib_levels if level < take_profit]
                if support_levels:
                    take_profit = max(support_levels)
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.debug(f"Error adjusting levels with Fibonacci: {e}")
            return stop_loss, take_profit
    
    def _validate_signals(self, signals: List[TradingSignal], data: pd.DataFrame) -> List[TradingSignal]:
        """
        Validate and filter trading signals.
        
        Args:
            signals: Raw signals
            data: OHLCV DataFrame
            
        Returns:
            Validated signals
        """
        validated_signals = []
        
        for signal in signals:
            # Filter by confidence
            if signal.confidence < self.min_confidence:
                continue
            
            # Check for conflicting signals
            if self._has_conflicting_signal(signal, validated_signals):
                continue
            
            # Validate risk/reward ratio
            if signal.stop_loss and signal.take_profit:
                risk = abs(signal.price - signal.stop_loss)
                reward = abs(signal.take_profit - signal.price)
                
                if risk > 0 and reward / risk < 1.5:  # Minimum 1.5:1 risk/reward
                    continue
            
            validated_signals.append(signal)
        
        return validated_signals
    
    def _has_conflicting_signal(self, signal: TradingSignal, existing_signals: List[TradingSignal]) -> bool:
        """
        Check if signal conflicts with existing signals.
        
        Args:
            signal: New signal
            existing_signals: Existing signals
            
        Returns:
            True if conflicting
        """
        for existing in existing_signals:
            # Same symbol and time frame
            if (existing.symbol == signal.symbol and 
                abs((existing.timestamp - signal.timestamp).days) < 1):
                
                # Conflicting directions
                if ((existing.signal_type in [SignalType.BUY] and signal.signal_type in [SignalType.SELL]) or
                    (existing.signal_type in [SignalType.SELL] and signal.signal_type in [SignalType.BUY])):
                    return True
        
        return False
    
    def update_positions(self, current_prices: Dict[str, float]):
        """
        Update current positions with latest prices.
        
        Args:
            current_prices: Dictionary of symbol -> current price
        """
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.current_price = current_prices[symbol]
    
    def get_position_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all positions.
        
        Returns:
            Dictionary of position statuses
        """
        status = {}
        
        for symbol, position in self.positions.items():
            status[symbol] = {
                'type': position.position_type.value,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'quantity': position.quantity,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_pct': position.unrealized_pnl_pct,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit
            }
        
        return status


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.data.data_loader import DataLoader
    
    # Load sample data
    loader = DataLoader()
    data = loader.get_yahoo_data("AAPL", period="1y")
    
    # Initialize strategy
    strategy = ElliottWaveStrategy()
    
    # Generate signals
    signals = strategy.generate_signals(data, "AAPL")
    
    print(f"Generated {len(signals)} trading signals for AAPL")
    
    # Print signal details
    for i, signal in enumerate(signals[:5]):  # Show first 5 signals
        print(f"\nSignal {i+1}:")
        print(f"  Time: {signal.timestamp}")
        print(f"  Type: {signal.signal_type.value}")
        print(f"  Price: {signal.price:.2f}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Wave: {signal.wave_type}")
        print(f"  Stop Loss: {signal.stop_loss:.2f}" if signal.stop_loss else "  Stop Loss: None")
        print(f"  Take Profit: {signal.take_profit:.2f}" if signal.take_profit else "  Take Profit: None")
        print(f"  Reason: {signal.reason}")
    
    # Get current wave analysis
    current_analysis = strategy.wave_detector.get_current_wave_count(data)
    print(f"\nCurrent Wave Analysis:")
    print(f"  Current Wave: {current_analysis['current_wave']}")
    print(f"  Confidence: {current_analysis['confidence']:.2f}")
    print(f"  Analysis: {current_analysis['analysis']}")
