"""
Fibonacci retracement and extension analysis for Elliott Wave Theory.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..utils.helpers import calculate_fibonacci_levels, calculate_fibonacci_extensions

logger = get_logger(__name__)


@dataclass
class FibonacciLevel:
    """Represents a Fibonacci level."""
    ratio: float
    price: float
    level_type: str  # 'retracement' or 'extension'
    is_key_level: bool = False
    
    def __str__(self):
        return f"Fib {self.ratio:.3f} at {self.price:.2f} ({self.level_type})"


@dataclass
class FibonacciAnalysis:
    """Complete Fibonacci analysis for a price move."""
    swing_high: float
    swing_low: float
    current_price: float
    retracements: List[FibonacciLevel]
    extensions: List[FibonacciLevel]
    key_levels: List[FibonacciLevel]
    trend_direction: str
    
    @property
    def nearest_support(self) -> Optional[FibonacciLevel]:
        """Get nearest support level below current price."""
        support_levels = [level for level in self.key_levels if level.price < self.current_price]
        return max(support_levels, key=lambda x: x.price) if support_levels else None
    
    @property
    def nearest_resistance(self) -> Optional[FibonacciLevel]:
        """Get nearest resistance level above current price."""
        resistance_levels = [level for level in self.key_levels if level.price > self.current_price]
        return min(resistance_levels, key=lambda x: x.price) if resistance_levels else None


class FibonacciAnalyzer:
    """
    Fibonacci retracement and extension analyzer for Elliott Wave Theory.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Fibonacci analyzer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.fibonacci_ratios = self.config.get('wave_detection.fibonacci_levels', 
                                               [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618])
        
        # Key Fibonacci levels that are most significant
        self.key_ratios = [0.382, 0.5, 0.618, 1.0, 1.618]
        
        # Extension ratios for projections
        self.extension_ratios = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618, 3.618, 4.236]
        
        logger.info("FibonacciAnalyzer initialized")
    
    def analyze_retracement(
        self, 
        swing_high: float, 
        swing_low: float, 
        current_price: float,
        trend_direction: str = 'up'
    ) -> FibonacciAnalysis:
        """
        Analyze Fibonacci retracement levels.
        
        Args:
            swing_high: High point of the move
            swing_low: Low point of the move
            current_price: Current market price
            trend_direction: 'up' or 'down'
            
        Returns:
            FibonacciAnalysis object
        """
        try:
            retracements = []
            extensions = []
            
            # Calculate retracement levels
            if trend_direction.lower() == 'up':
                # Uptrend: retracements from high to low
                fib_levels = calculate_fibonacci_levels(swing_high, swing_low, 'uptrend')
            else:
                # Downtrend: retracements from low to high
                fib_levels = calculate_fibonacci_levels(swing_low, swing_high, 'downtrend')
            
            # Create FibonacciLevel objects for retracements
            for ratio_str, price in fib_levels.items():
                ratio = float(ratio_str.replace('fib_', ''))
                is_key = ratio in self.key_ratios
                
                level = FibonacciLevel(
                    ratio=ratio,
                    price=price,
                    level_type='retracement',
                    is_key_level=is_key
                )
                retracements.append(level)
            
            # Calculate extension levels
            extension_levels = self._calculate_extensions(swing_high, swing_low, trend_direction)
            for ratio, price in extension_levels.items():
                is_key = ratio in self.key_ratios
                
                level = FibonacciLevel(
                    ratio=ratio,
                    price=price,
                    level_type='extension',
                    is_key_level=is_key
                )
                extensions.append(level)
            
            # Combine all levels and identify key ones
            all_levels = retracements + extensions
            key_levels = [level for level in all_levels if level.is_key_level]
            
            analysis = FibonacciAnalysis(
                swing_high=swing_high,
                swing_low=swing_low,
                current_price=current_price,
                retracements=retracements,
                extensions=extensions,
                key_levels=key_levels,
                trend_direction=trend_direction
            )
            
            logger.debug(f"Fibonacci analysis complete: {len(retracements)} retracements, {len(extensions)} extensions")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in Fibonacci analysis: {e}")
            raise
    
    def _calculate_extensions(self, swing_high: float, swing_low: float, trend_direction: str) -> Dict[float, float]:
        """
        Calculate Fibonacci extension levels.
        
        Args:
            swing_high: High point
            swing_low: Low point
            trend_direction: Trend direction
            
        Returns:
            Dictionary of ratio -> price
        """
        extensions = {}
        wave_height = abs(swing_high - swing_low)
        
        if trend_direction.lower() == 'up':
            # Extensions above the high
            base_price = swing_high
            for ratio in self.extension_ratios:
                extensions[ratio] = base_price + wave_height * ratio
        else:
            # Extensions below the low
            base_price = swing_low
            for ratio in self.extension_ratios:
                extensions[ratio] = base_price - wave_height * ratio
        
        return extensions
    
    def find_confluence_zones(self, analyses: List[FibonacciAnalysis], tolerance: float = 0.01) -> List[Dict[str, Any]]:
        """
        Find confluence zones where multiple Fibonacci levels cluster.
        
        Args:
            analyses: List of FibonacciAnalysis objects
            tolerance: Price tolerance for clustering (as percentage)
            
        Returns:
            List of confluence zones
        """
        confluence_zones = []
        all_levels = []
        
        # Collect all levels from all analyses
        for analysis in analyses:
            all_levels.extend(analysis.key_levels)
        
        if not all_levels:
            return confluence_zones
        
        # Sort levels by price
        all_levels.sort(key=lambda x: x.price)
        
        # Find clusters of levels
        current_cluster = [all_levels[0]]
        
        for level in all_levels[1:]:
            # Check if this level is close to the current cluster
            cluster_avg_price = sum(l.price for l in current_cluster) / len(current_cluster)
            
            if abs(level.price - cluster_avg_price) / cluster_avg_price <= tolerance:
                current_cluster.append(level)
            else:
                # Save current cluster if it has multiple levels
                if len(current_cluster) >= 2:
                    confluence_zones.append(self._create_confluence_zone(current_cluster))
                
                # Start new cluster
                current_cluster = [level]
        
        # Don't forget the last cluster
        if len(current_cluster) >= 2:
            confluence_zones.append(self._create_confluence_zone(current_cluster))
        
        return confluence_zones
    
    def _create_confluence_zone(self, levels: List[FibonacciLevel]) -> Dict[str, Any]:
        """
        Create a confluence zone from clustered levels.
        
        Args:
            levels: List of clustered levels
            
        Returns:
            Confluence zone dictionary
        """
        avg_price = sum(level.price for level in levels) / len(levels)
        min_price = min(level.price for level in levels)
        max_price = max(level.price for level in levels)
        
        ratios = [level.ratio for level in levels]
        level_types = [level.level_type for level in levels]
        
        strength = len(levels)  # More levels = stronger confluence
        
        return {
            'price': avg_price,
            'price_range': (min_price, max_price),
            'strength': strength,
            'ratios': ratios,
            'level_types': level_types,
            'levels': levels
        }
    
    def analyze_price_at_fibonacci(self, current_price: float, analysis: FibonacciAnalysis) -> Dict[str, Any]:
        """
        Analyze current price position relative to Fibonacci levels.
        
        Args:
            current_price: Current market price
            analysis: FibonacciAnalysis object
            
        Returns:
            Dictionary with price analysis
        """
        nearest_levels = []
        tolerance = 0.005  # 0.5% tolerance
        
        # Find levels close to current price
        for level in analysis.key_levels:
            price_diff_pct = abs(current_price - level.price) / current_price
            if price_diff_pct <= tolerance:
                nearest_levels.append({
                    'level': level,
                    'distance_pct': price_diff_pct
                })
        
        # Sort by distance
        nearest_levels.sort(key=lambda x: x['distance_pct'])
        
        # Determine support and resistance
        support_levels = [level for level in analysis.key_levels if level.price < current_price]
        resistance_levels = [level for level in analysis.key_levels if level.price > current_price]
        
        next_support = max(support_levels, key=lambda x: x.price) if support_levels else None
        next_resistance = min(resistance_levels, key=lambda x: x.price) if resistance_levels else None
        
        return {
            'current_price': current_price,
            'at_fibonacci_level': len(nearest_levels) > 0,
            'nearest_levels': nearest_levels,
            'next_support': next_support,
            'next_resistance': next_resistance,
            'price_position': self._get_price_position(current_price, analysis)
        }
    
    def _get_price_position(self, current_price: float, analysis: FibonacciAnalysis) -> str:
        """
        Get descriptive position of price relative to Fibonacci levels.
        
        Args:
            current_price: Current price
            analysis: FibonacciAnalysis object
            
        Returns:
            Descriptive position string
        """
        if analysis.trend_direction.lower() == 'up':
            # In uptrend
            if current_price > analysis.swing_high:
                return "Above swing high - potential extension"
            elif current_price >= analysis.swing_high * 0.95:
                return "Near swing high"
            elif current_price >= analysis.swing_low + (analysis.swing_high - analysis.swing_low) * 0.618:
                return "Above 61.8% retracement"
            elif current_price >= analysis.swing_low + (analysis.swing_high - analysis.swing_low) * 0.5:
                return "Above 50% retracement"
            elif current_price >= analysis.swing_low + (analysis.swing_high - analysis.swing_low) * 0.382:
                return "Above 38.2% retracement"
            else:
                return "Deep retracement"
        else:
            # In downtrend
            if current_price < analysis.swing_low:
                return "Below swing low - potential extension"
            elif current_price <= analysis.swing_low * 1.05:
                return "Near swing low"
            elif current_price <= analysis.swing_high - (analysis.swing_high - analysis.swing_low) * 0.618:
                return "Below 61.8% retracement"
            elif current_price <= analysis.swing_high - (analysis.swing_high - analysis.swing_low) * 0.5:
                return "Below 50% retracement"
            elif current_price <= analysis.swing_high - (analysis.swing_high - analysis.swing_low) * 0.382:
                return "Below 38.2% retracement"
            else:
                return "Deep retracement"
    
    def calculate_wave_targets(
        self, 
        wave_start: float, 
        wave_end: float, 
        wave_type: str,
        reference_wave_start: float = None,
        reference_wave_end: float = None
    ) -> Dict[str, float]:
        """
        Calculate potential targets for Elliott Waves using Fibonacci ratios.
        
        Args:
            wave_start: Starting price of current wave
            wave_end: Current price (end of current wave so far)
            wave_type: Type of wave (1, 2, 3, 4, 5, A, B, C)
            reference_wave_start: Start of reference wave for projections
            reference_wave_end: End of reference wave for projections
            
        Returns:
            Dictionary of target levels
        """
        targets = {}
        
        try:
            if wave_type in ['3', 'C']:
                # Wave 3 and C targets based on Wave 1/A
                if reference_wave_start is not None and reference_wave_end is not None:
                    reference_length = abs(reference_wave_end - reference_wave_start)
                    
                    # Common ratios for Wave 3
                    if wave_type == '3':
                        ratios = [1.618, 2.618, 4.236]
                    else:  # Wave C
                        ratios = [0.618, 1.0, 1.618]
                    
                    direction = 1 if wave_end > wave_start else -1
                    
                    for ratio in ratios:
                        target_price = wave_start + direction * reference_length * ratio
                        targets[f"target_{ratio}"] = target_price
            
            elif wave_type in ['5']:
                # Wave 5 targets
                if reference_wave_start is not None and reference_wave_end is not None:
                    wave_1_length = abs(reference_wave_end - reference_wave_start)
                    
                    # Wave 5 is often equal to Wave 1 or 0.618 of Wave 1
                    ratios = [0.618, 1.0, 1.618]
                    direction = 1 if wave_end > wave_start else -1
                    
                    for ratio in ratios:
                        target_price = wave_start + direction * wave_1_length * ratio
                        targets[f"target_{ratio}"] = target_price
            
            elif wave_type in ['2', '4', 'B']:
                # Retracement targets
                previous_wave_length = abs(wave_start - (reference_wave_start or wave_start))
                
                # Common retracement ratios
                ratios = [0.382, 0.5, 0.618, 0.786]
                direction = -1 if wave_end > wave_start else 1  # Opposite direction for retracements
                
                for ratio in ratios:
                    target_price = wave_start + direction * previous_wave_length * ratio
                    targets[f"retracement_{ratio}"] = target_price
            
            logger.debug(f"Calculated {len(targets)} targets for wave {wave_type}")
            return targets
            
        except Exception as e:
            logger.error(f"Error calculating wave targets: {e}")
            return {}
    
    def get_fibonacci_confluence(
        self, 
        data: pd.DataFrame, 
        lookback_periods: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Find Fibonacci confluence zones in recent price action.
        
        Args:
            data: OHLCV DataFrame
            lookback_periods: Number of periods to analyze
            
        Returns:
            List of confluence zones
        """
        try:
            recent_data = data.tail(lookback_periods)
            analyses = []
            
            # Find significant swings in recent data
            from ..data.indicators import TechnicalIndicators
            
            # Get swing points
            swings = TechnicalIndicators.swing_points(recent_data, threshold=0.02)
            
            # Analyze each swing
            for i in range(1, len(swings)):
                prev_swing = swings[i-1]
                curr_swing = swings[i]
                
                if prev_swing['type'] != curr_swing['type']:  # Different swing types
                    high_price = max(prev_swing['price'], curr_swing['price'])
                    low_price = min(prev_swing['price'], curr_swing['price'])
                    
                    trend_dir = 'up' if curr_swing['type'] == 'high' else 'down'
                    
                    analysis = self.analyze_retracement(
                        high_price, 
                        low_price, 
                        recent_data['close'].iloc[-1],
                        trend_dir
                    )
                    
                    analyses.append(analysis)
            
            # Find confluence zones
            confluence_zones = self.find_confluence_zones(analyses)
            
            logger.info(f"Found {len(confluence_zones)} Fibonacci confluence zones")
            return confluence_zones
            
        except Exception as e:
            logger.error(f"Error finding Fibonacci confluence: {e}")
            return []


if __name__ == "__main__":
    # Example usage
    analyzer = FibonacciAnalyzer()
    
    # Example retracement analysis
    analysis = analyzer.analyze_retracement(
        swing_high=150.0,
        swing_low=100.0,
        current_price=125.0,
        trend_direction='up'
    )
    
    print(f"Fibonacci Analysis:")
    print(f"Swing High: {analysis.swing_high}")
    print(f"Swing Low: {analysis.swing_low}")
    print(f"Current Price: {analysis.current_price}")
    print(f"Trend Direction: {analysis.trend_direction}")
    
    print(f"\nKey Retracement Levels:")
    for level in analysis.retracements:
        if level.is_key_level:
            print(f"  {level}")
    
    print(f"\nKey Extension Levels:")
    for level in analysis.extensions:
        if level.is_key_level:
            print(f"  {level}")
    
    # Price analysis
    price_analysis = analyzer.analyze_price_at_fibonacci(125.0, analysis)
    print(f"\nPrice Analysis: {price_analysis['price_position']}")
    
    if price_analysis['next_support']:
        print(f"Next Support: {price_analysis['next_support']}")
    if price_analysis['next_resistance']:
        print(f"Next Resistance: {price_analysis['next_resistance']}")
    
    # Wave targets example
    targets = analyzer.calculate_wave_targets(
        wave_start=100.0,
        wave_end=125.0,
        wave_type='3',
        reference_wave_start=100.0,
        reference_wave_end=120.0
    )
    
    print(f"\nWave 3 Targets:")
    for target_name, target_price in targets.items():
        print(f"  {target_name}: {target_price:.2f}")
