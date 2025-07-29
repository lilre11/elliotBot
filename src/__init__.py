"""
Elliott Wave Trading Bot Package
"""

__version__ = "1.0.0"
__author__ = "Elliott Wave Trading Bot"
__description__ = "Advanced AI-powered trading bot using Elliott Wave Theory"

from .utils.config import get_config
from .utils.logger import get_logger, TradingLogger

__all__ = ['get_config', 'get_logger', 'TradingLogger']
