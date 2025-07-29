"""
Logging utilities for Elliott Wave Trading Bot.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file
        log_level: Logging level
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    logger = logging.getLogger('elliott_bot')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        
    Returns:
        Logger instance
    """
    # Use the module-specific logger as a child of the main logger
    logger_name = f'elliott_bot.{name}' if not name.startswith('elliott_bot') else name
    logger = logging.getLogger(logger_name)
    
    # If no handlers exist, set up basic logging
    if not logger.handlers and not logger.parent.handlers:
        setup_logging(log_file)
    
    return logger


class TradingLogger:
    """
    Specialized logger for trading operations.
    """
    
    def __init__(self, name: str = "trading"):
        self.logger = get_logger(name)
        self.trade_log_file = "logs/trades.log"
        self.signal_log_file = "logs/signals.log"
        
        # Set up specialized loggers
        self._setup_trade_logger()
        self._setup_signal_logger()
    
    def _setup_trade_logger(self):
        """Set up trade-specific logging."""
        Path(self.trade_log_file).parent.mkdir(parents=True, exist_ok=True)
        
        trade_formatter = logging.Formatter(
            '%(asctime)s,%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.trade_logger = logging.getLogger('elliott_bot.trades')
        self.trade_logger.setLevel(logging.INFO)
        
        trade_handler = logging.handlers.RotatingFileHandler(
            self.trade_log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=10
        )
        trade_handler.setFormatter(trade_formatter)
        self.trade_logger.addHandler(trade_handler)
    
    def _setup_signal_logger(self):
        """Set up signal-specific logging."""
        Path(self.signal_log_file).parent.mkdir(parents=True, exist_ok=True)
        
        signal_formatter = logging.Formatter(
            '%(asctime)s,%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.signal_logger = logging.getLogger('elliott_bot.signals')
        self.signal_logger.setLevel(logging.INFO)
        
        signal_handler = logging.handlers.RotatingFileHandler(
            self.signal_log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=10
        )
        signal_handler.setFormatter(signal_formatter)
        self.signal_logger.addHandler(signal_handler)
    
    def log_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        timestamp: str,
        strategy: str = "",
        confidence: float = 0.0
    ):
        """
        Log a trade execution.
        
        Args:
            symbol: Trading symbol
            action: Trade action (BUY/SELL)
            quantity: Trade quantity
            price: Execution price
            timestamp: Trade timestamp
            strategy: Strategy name
            confidence: Signal confidence
        """
        trade_msg = f"{symbol},{action},{quantity},{price},{strategy},{confidence}"
        self.trade_logger.info(trade_msg)
        
        # Also log to main logger
        self.logger.info(f"Trade executed: {action} {quantity} {symbol} @ {price}")
    
    def log_signal(
        self,
        symbol: str,
        signal_type: str,
        wave_count: str,
        confidence: float,
        price: float,
        timestamp: str,
        fibonacci_level: float = 0.0
    ):
        """
        Log a trading signal.
        
        Args:
            symbol: Trading symbol
            signal_type: Signal type (BUY/SELL/HOLD)
            wave_count: Current wave count
            confidence: Signal confidence
            price: Current price
            timestamp: Signal timestamp
            fibonacci_level: Relevant Fibonacci level
        """
        signal_msg = f"{symbol},{signal_type},{wave_count},{confidence},{price},{fibonacci_level}"
        self.signal_logger.info(signal_msg)
        
        # Also log to main logger
        self.logger.info(f"Signal: {signal_type} {symbol} at {price} (Wave: {wave_count}, Confidence: {confidence:.2f})")
    
    def log_wave_detection(
        self,
        symbol: str,
        wave_pattern: str,
        confidence: float,
        start_time: str,
        end_time: str,
        price_range: tuple
    ):
        """
        Log wave pattern detection.
        
        Args:
            symbol: Trading symbol
            wave_pattern: Detected wave pattern
            confidence: Detection confidence
            start_time: Wave start time
            end_time: Wave end time
            price_range: (start_price, end_price)
        """
        self.logger.info(
            f"Wave detected: {symbol} - {wave_pattern} "
            f"({start_time} to {end_time}) "
            f"Price: {price_range[0]:.2f} -> {price_range[1]:.2f} "
            f"Confidence: {confidence:.2f}"
        )
    
    def log_backtest_result(
        self,
        strategy: str,
        total_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float,
        total_trades: int
    ):
        """
        Log backtesting results.
        
        Args:
            strategy: Strategy name
            total_return: Total return percentage
            sharpe_ratio: Sharpe ratio
            max_drawdown: Maximum drawdown
            win_rate: Win rate percentage
            total_trades: Total number of trades
        """
        self.logger.info(
            f"Backtest complete - {strategy}: "
            f"Return: {total_return:.2f}%, "
            f"Sharpe: {sharpe_ratio:.2f}, "
            f"Max DD: {max_drawdown:.2f}%, "
            f"Win Rate: {win_rate:.2f}%, "
            f"Trades: {total_trades}"
        )


# Initialize logging on module import
try:
    from .config import get_config
    config = get_config()
    log_file = config.get('logging.file_path', 'logs/elliott_bot.log')
    log_level = config.get('logging.level', 'INFO')
    setup_logging(log_file, log_level)
except Exception:
    # Fallback to basic logging if config is not available
    setup_logging()


if __name__ == "__main__":
    # Example usage
    logger = get_logger(__name__)
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Trading logger example
    trading_logger = TradingLogger()
    trading_logger.log_trade(
        symbol="AAPL",
        action="BUY",
        quantity=100,
        price=150.25,
        timestamp="2024-01-01 10:30:00",
        strategy="Elliott Wave",
        confidence=0.85
    )
    
    trading_logger.log_signal(
        symbol="AAPL",
        signal_type="BUY",
        wave_count="Wave 3",
        confidence=0.85,
        price=150.25,
        timestamp="2024-01-01 10:30:00",
        fibonacci_level=0.618
    )
