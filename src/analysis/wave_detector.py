"""
Elliott Wave detection and labeling module.
Implements pattern recognition for identifying Elliott Wave structures.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
import logging
from scipy.signal import find_peaks

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..utils.helpers import calculate_fibonacci_levels, find_peaks_and_troughs

logger = get_logger(__name__)


class WaveType(Enum):
    """Elliott Wave types."""
    IMPULSE_1 = "1"
    IMPULSE_2 = "2"
    IMPULSE_3 = "3"
    IMPULSE_4 = "4"
    IMPULSE_5 = "5"
    CORRECTIVE_A = "A"
    CORRECTIVE_B = "B"
    CORRECTIVE_C = "C"
    UNKNOWN = "?"


class TrendDirection(Enum):
    """Trend direction."""
    UP = 1
    DOWN = -1
    SIDEWAYS = 0


@dataclass
class WavePoint:
    """Represents a point in Elliott Wave analysis."""
    timestamp: pd.Timestamp
    price: float
    index: int
    wave_type: WaveType
    confidence: float = 0.0
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = pd.Timestamp(self.timestamp)


@dataclass
class Wave:
    """Represents an Elliott Wave."""
    start_point: WavePoint
    end_point: WavePoint
    wave_type: WaveType
    direction: TrendDirection
    confidence: float
    fibonacci_ratios: Dict[str, float] = None
    
    def __post_init__(self):
        if self.fibonacci_ratios is None:
            self.fibonacci_ratios = {}
    
    @property
    def duration(self) -> int:
        """Get wave duration in periods."""
        return self.end_point.index - self.start_point.index
    
    @property
    def price_change(self) -> float:
        """Get wave price change."""
        return self.end_point.price - self.start_point.price
    
    @property
    def price_change_pct(self) -> float:
        """Get wave price change percentage."""
        return (self.end_point.price - self.start_point.price) / self.start_point.price


class WaveDetector:
    """
    Main Elliott Wave detection and analysis class.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize WaveDetector.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.zigzag_threshold = self.config.get('wave_detection.zigzag_threshold', 0.05)
        self.min_wave_length = self.config.get('wave_detection.min_wave_length', 5)
        self.fibonacci_levels = self.config.get('wave_detection.fibonacci_levels', 
                                               [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618])
        self.confidence_threshold = self.config.get('wave_detection.confidence_threshold', 0.7)
        
        logger.info(f"WaveDetector initialized with threshold: {self.zigzag_threshold}")
    
    def detect_waves(self, data: pd.DataFrame) -> List[Wave]:
        """
        Detect Elliott Waves in price data.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            List of detected waves
        """
        try:
            # Step 1: Identify swing points using ZigZag
            swing_points = self._get_swing_points(data)
            
            if len(swing_points) < 5:
                logger.warning("Insufficient swing points for wave analysis")
                return []
            
            # Step 2: Analyze wave patterns
            waves = self._analyze_wave_patterns(swing_points, data)
            
            # Step 3: Validate and score waves
            validated_waves = self._validate_waves(waves, data)
            
            logger.info(f"Detected {len(validated_waves)} valid Elliott Waves")
            return validated_waves
            
        except Exception as e:
            logger.error(f"Error detecting waves: {e}")
            return []
    
    def _get_swing_points(self, data: pd.DataFrame) -> List[WavePoint]:
        """
        Identify swing points using ZigZag indicator.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            List of swing points
        """
        from ..data.indicators import TechnicalIndicators
        
        # Calculate ZigZag
        zigzag, direction = TechnicalIndicators.zigzag(data, self.zigzag_threshold)
        
        swing_points = []
        
        for timestamp, price in zigzag.dropna().items():
            if pd.isna(price):
                continue
                
            index = data.index.get_loc(timestamp)
            wave_direction = direction.loc[timestamp]
            
            # Create wave point
            point = WavePoint(
                timestamp=timestamp,
                price=price,
                index=index,
                wave_type=WaveType.UNKNOWN,
                confidence=1.0  # Initial confidence
            )
            
            swing_points.append(point)
        
        # Sort by timestamp
        swing_points.sort(key=lambda x: x.timestamp)
        
        logger.debug(f"Found {len(swing_points)} swing points")
        return swing_points
    
    def _analyze_wave_patterns(self, swing_points: List[WavePoint], data: pd.DataFrame) -> List[Wave]:
        """
        Analyze swing points to identify Elliott Wave patterns.
        
        Args:
            swing_points: List of swing points
            data: OHLCV DataFrame
            
        Returns:
            List of identified waves
        """
        waves = []
        
        # Look for 5-wave impulse patterns
        impulse_waves = self._find_impulse_patterns(swing_points, data)
        waves.extend(impulse_waves)
        
        # Look for 3-wave corrective patterns
        corrective_waves = self._find_corrective_patterns(swing_points, data)
        waves.extend(corrective_waves)
        
        return waves
    
    def _find_impulse_patterns(self, swing_points: List[WavePoint], data: pd.DataFrame) -> List[Wave]:
        """
        Find 5-wave impulse patterns (1-2-3-4-5).
        
        Args:
            swing_points: List of swing points
            data: OHLCV DataFrame
            
        Returns:
            List of impulse waves
        """
        impulse_waves = []
        
        # Need at least 6 points for a complete 5-wave pattern
        if len(swing_points) < 6:
            return impulse_waves
        
        # Sliding window to find 5-wave patterns
        for i in range(len(swing_points) - 5):
            pattern_points = swing_points[i:i+6]  # 6 points define 5 waves
            
            # Check if this could be an impulse pattern
            if self._validate_impulse_pattern(pattern_points, data):
                # Create individual waves
                wave_types = [WaveType.IMPULSE_1, WaveType.IMPULSE_2, WaveType.IMPULSE_3, 
                             WaveType.IMPULSE_4, WaveType.IMPULSE_5]
                
                for j in range(5):
                    start_point = pattern_points[j]
                    end_point = pattern_points[j+1]
                    
                    # Determine direction
                    direction = TrendDirection.UP if end_point.price > start_point.price else TrendDirection.DOWN
                    
                    # Calculate confidence
                    confidence = self._calculate_wave_confidence(start_point, end_point, wave_types[j], data)
                    
                    # Create wave
                    wave = Wave(
                        start_point=start_point,
                        end_point=end_point,
                        wave_type=wave_types[j],
                        direction=direction,
                        confidence=confidence
                    )
                    
                    # Add Fibonacci analysis
                    wave.fibonacci_ratios = self._calculate_fibonacci_ratios(wave, pattern_points)
                    
                    impulse_waves.append(wave)
        
        return impulse_waves
    
    def _find_corrective_patterns(self, swing_points: List[WavePoint], data: pd.DataFrame) -> List[Wave]:
        """
        Find 3-wave corrective patterns (A-B-C).
        
        Args:
            swing_points: List of swing points
            data: OHLCV DataFrame
            
        Returns:
            List of corrective waves
        """
        corrective_waves = []
        
        # Need at least 4 points for a complete 3-wave pattern
        if len(swing_points) < 4:
            return corrective_waves
        
        # Sliding window to find 3-wave patterns
        for i in range(len(swing_points) - 3):
            pattern_points = swing_points[i:i+4]  # 4 points define 3 waves
            
            # Check if this could be a corrective pattern
            if self._validate_corrective_pattern(pattern_points, data):
                # Create individual waves
                wave_types = [WaveType.CORRECTIVE_A, WaveType.CORRECTIVE_B, WaveType.CORRECTIVE_C]
                
                for j in range(3):
                    start_point = pattern_points[j]
                    end_point = pattern_points[j+1]
                    
                    # Determine direction
                    direction = TrendDirection.UP if end_point.price > start_point.price else TrendDirection.DOWN
                    
                    # Calculate confidence
                    confidence = self._calculate_wave_confidence(start_point, end_point, wave_types[j], data)
                    
                    # Create wave
                    wave = Wave(
                        start_point=start_point,
                        end_point=end_point,
                        wave_type=wave_types[j],
                        direction=direction,
                        confidence=confidence
                    )
                    
                    # Add Fibonacci analysis
                    wave.fibonacci_ratios = self._calculate_fibonacci_ratios(wave, pattern_points)
                    
                    corrective_waves.append(wave)
        
        return corrective_waves
    
    def _validate_impulse_pattern(self, points: List[WavePoint], data: pd.DataFrame) -> bool:
        """
        Validate if points form a valid 5-wave impulse pattern.
        
        Args:
            points: List of 6 points defining 5 waves
            data: OHLCV DataFrame
            
        Returns:
            True if valid impulse pattern
        """
        if len(points) != 6:
            return False
        
        # Elliott Wave rules for impulse patterns:
        # 1. Wave 2 cannot retrace more than 100% of wave 1
        # 2. Wave 3 cannot be the shortest wave among waves 1, 3, and 5
        # 3. Wave 4 cannot overlap with wave 1 (except in diagonal patterns)
        
        try:
            # Calculate wave lengths
            wave1_len = abs(points[1].price - points[0].price)
            wave2_len = abs(points[2].price - points[1].price)
            wave3_len = abs(points[3].price - points[2].price)
            wave4_len = abs(points[4].price - points[3].price)
            wave5_len = abs(points[5].price - points[4].price)
            
            # Rule 1: Wave 2 retracement
            wave2_retracement = wave2_len / wave1_len
            if wave2_retracement > 1.0:
                return False
            
            # Rule 2: Wave 3 length
            if wave3_len <= min(wave1_len, wave5_len):
                return False
            
            # Rule 3: Wave 4 overlap (simplified check)
            if points[0].price < points[4].price < points[1].price or points[1].price < points[4].price < points[0].price:
                return False
            
            # Additional checks
            # Minimum wave length
            for i in range(1, 6):
                wave_duration = points[i].index - points[i-1].index
                if wave_duration < self.min_wave_length:
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error validating impulse pattern: {e}")
            return False
    
    def _validate_corrective_pattern(self, points: List[WavePoint], data: pd.DataFrame) -> bool:
        """
        Validate if points form a valid 3-wave corrective pattern.
        
        Args:
            points: List of 4 points defining 3 waves
            data: OHLCV DataFrame
            
        Returns:
            True if valid corrective pattern
        """
        if len(points) != 4:
            return False
        
        try:
            # Basic validation for corrective patterns
            # Minimum wave length
            for i in range(1, 4):
                wave_duration = points[i].index - points[i-1].index
                if wave_duration < self.min_wave_length:
                    return False
            
            # Wave C should typically be related to wave A by Fibonacci ratios
            wave_a_len = abs(points[1].price - points[0].price)
            wave_c_len = abs(points[3].price - points[2].price)
            
            if wave_a_len > 0:
                c_to_a_ratio = wave_c_len / wave_a_len
                # Common Fibonacci ratios for wave C
                valid_ratios = [0.618, 1.0, 1.618, 2.618]
                tolerance = 0.1
                
                ratio_valid = any(abs(c_to_a_ratio - ratio) / ratio <= tolerance for ratio in valid_ratios)
                if not ratio_valid:
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error validating corrective pattern: {e}")
            return False
    
    def _calculate_wave_confidence(
        self, 
        start_point: WavePoint, 
        end_point: WavePoint, 
        wave_type: WaveType, 
        data: pd.DataFrame
    ) -> float:
        """
        Calculate confidence score for a wave.
        
        Args:
            start_point: Wave start point
            end_point: Wave end point
            wave_type: Type of wave
            data: OHLCV DataFrame
            
        Returns:
            Confidence score (0-1)
        """
        confidence_factors = []
        
        try:
            # Factor 1: Wave length relative to threshold
            price_change_pct = abs(end_point.price - start_point.price) / start_point.price
            length_factor = min(price_change_pct / self.zigzag_threshold, 1.0)
            confidence_factors.append(length_factor)
            
            # Factor 2: Volume confirmation
            wave_data = data.iloc[start_point.index:end_point.index+1]
            if len(wave_data) > 1:
                avg_volume = wave_data['volume'].mean()
                baseline_volume = data['volume'].rolling(20).mean().iloc[end_point.index]
                volume_factor = min(avg_volume / baseline_volume, 1.5) / 1.5
                confidence_factors.append(volume_factor)
            
            # Factor 3: Wave type specific rules
            type_factor = self._get_wave_type_confidence(wave_type, start_point, end_point)
            confidence_factors.append(type_factor)
            
            # Factor 4: Duration factor
            duration = end_point.index - start_point.index
            duration_factor = min(duration / (self.min_wave_length * 2), 1.0)
            confidence_factors.append(duration_factor)
            
            # Calculate weighted average
            weights = [0.3, 0.25, 0.25, 0.2]  # Adjust weights as needed
            confidence = sum(factor * weight for factor, weight in zip(confidence_factors, weights))
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.debug(f"Error calculating wave confidence: {e}")
            return 0.5  # Default confidence
    
    def _get_wave_type_confidence(self, wave_type: WaveType, start_point: WavePoint, end_point: WavePoint) -> float:
        """
        Get confidence factor based on wave type specific characteristics.
        
        Args:
            wave_type: Type of wave
            start_point: Wave start point
            end_point: Wave end point
            
        Returns:
            Type-specific confidence factor
        """
        # Simplified type-specific confidence
        if wave_type in [WaveType.IMPULSE_1, WaveType.IMPULSE_3, WaveType.IMPULSE_5]:
            # Impulse waves should be strong moves
            return 0.8
        elif wave_type in [WaveType.IMPULSE_2, WaveType.IMPULSE_4]:
            # Corrective waves within impulse should be smaller
            return 0.7
        elif wave_type in [WaveType.CORRECTIVE_A, WaveType.CORRECTIVE_C]:
            # Corrective waves
            return 0.6
        elif wave_type == WaveType.CORRECTIVE_B:
            # B waves are often irregular
            return 0.5
        else:
            return 0.5
    
    def _calculate_fibonacci_ratios(self, wave: Wave, pattern_points: List[WavePoint]) -> Dict[str, float]:
        """
        Calculate Fibonacci ratios for the wave relative to the pattern.
        
        Args:
            wave: Wave to analyze
            pattern_points: Points defining the pattern
            
        Returns:
            Dictionary of Fibonacci ratios
        """
        ratios = {}
        
        try:
            if len(pattern_points) >= 3:
                # Calculate retracement levels
                if wave.wave_type in [WaveType.IMPULSE_2, WaveType.IMPULSE_4, WaveType.CORRECTIVE_B]:
                    # For retracement waves
                    previous_wave_start = pattern_points[0].price
                    previous_wave_end = pattern_points[1].price
                    current_level = wave.end_point.price
                    
                    if previous_wave_end != previous_wave_start:
                        retracement = abs(current_level - previous_wave_end) / abs(previous_wave_end - previous_wave_start)
                        ratios['retracement'] = retracement
                        
                        # Find closest Fibonacci level
                        for fib_level in self.fibonacci_levels:
                            if abs(retracement - fib_level) < 0.05:  # 5% tolerance
                                ratios['fibonacci_match'] = fib_level
                                break
                
                # Calculate extension levels for impulse waves
                elif wave.wave_type in [WaveType.IMPULSE_3, WaveType.IMPULSE_5]:
                    if len(pattern_points) >= 4:
                        wave1_length = abs(pattern_points[1].price - pattern_points[0].price)
                        current_wave_length = abs(wave.end_point.price - wave.start_point.price)
                        
                        if wave1_length > 0:
                            extension_ratio = current_wave_length / wave1_length
                            ratios['extension'] = extension_ratio
                            
                            # Find closest Fibonacci extension
                            extension_levels = [1.0, 1.618, 2.618, 4.236]
                            for ext_level in extension_levels:
                                if abs(extension_ratio - ext_level) < 0.1:  # 10% tolerance
                                    ratios['fibonacci_extension'] = ext_level
                                    break
        
        except Exception as e:
            logger.debug(f"Error calculating Fibonacci ratios: {e}")
        
        return ratios
    
    def _validate_waves(self, waves: List[Wave], data: pd.DataFrame) -> List[Wave]:
        """
        Validate and filter waves based on confidence and rules.
        
        Args:
            waves: List of detected waves
            data: OHLCV DataFrame
            
        Returns:
            List of validated waves
        """
        validated_waves = []
        
        for wave in waves:
            # Filter by confidence threshold
            if wave.confidence >= self.confidence_threshold:
                validated_waves.append(wave)
        
        # Remove overlapping waves (keep higher confidence)
        validated_waves = self._remove_overlapping_waves(validated_waves)
        
        # Sort by timestamp
        validated_waves.sort(key=lambda w: w.start_point.timestamp)
        
        return validated_waves
    
    def _remove_overlapping_waves(self, waves: List[Wave]) -> List[Wave]:
        """
        Remove overlapping waves, keeping those with higher confidence.
        
        Args:
            waves: List of waves
            
        Returns:
            List of non-overlapping waves
        """
        if not waves:
            return waves
        
        # Sort by confidence (descending)
        sorted_waves = sorted(waves, key=lambda w: w.confidence, reverse=True)
        
        non_overlapping = []
        
        for wave in sorted_waves:
            overlaps = False
            
            for existing_wave in non_overlapping:
                # Check for overlap
                if (wave.start_point.index <= existing_wave.end_point.index and 
                    wave.end_point.index >= existing_wave.start_point.index):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(wave)
        
        return non_overlapping
    
    def get_current_wave_count(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get the current wave count and analysis.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dictionary with current wave analysis
        """
        waves = self.detect_waves(data)
        
        if not waves:
            return {
                'current_wave': 'Unknown',
                'confidence': 0.0,
                'next_target': None,
                'analysis': 'Insufficient data for wave analysis'
            }
        
        # Get the most recent wave
        latest_wave = max(waves, key=lambda w: w.end_point.timestamp)
        
        # Determine what might come next
        next_target = self._predict_next_wave(latest_wave, waves)
        
        return {
            'current_wave': latest_wave.wave_type.value,
            'confidence': latest_wave.confidence,
            'direction': latest_wave.direction.name,
            'next_target': next_target,
            'fibonacci_ratios': latest_wave.fibonacci_ratios,
            'analysis': f"Currently in {latest_wave.wave_type.value} wave with {latest_wave.confidence:.2f} confidence"
        }
    
    def _predict_next_wave(self, current_wave: Wave, all_waves: List[Wave]) -> Optional[Dict[str, Any]]:
        """
        Predict the next likely wave based on Elliott Wave theory.
        
        Args:
            current_wave: Current wave
            all_waves: All detected waves
            
        Returns:
            Dictionary with next wave prediction
        """
        try:
            if current_wave.wave_type == WaveType.IMPULSE_1:
                return {'type': 'Corrective Wave 2', 'direction': 'Opposite to Wave 1'}
            elif current_wave.wave_type == WaveType.IMPULSE_2:
                return {'type': 'Impulse Wave 3', 'direction': 'Same as Wave 1, likely extended'}
            elif current_wave.wave_type == WaveType.IMPULSE_3:
                return {'type': 'Corrective Wave 4', 'direction': 'Opposite to Wave 3'}
            elif current_wave.wave_type == WaveType.IMPULSE_4:
                return {'type': 'Impulse Wave 5', 'direction': 'Same as Wave 3'}
            elif current_wave.wave_type == WaveType.IMPULSE_5:
                return {'type': 'Corrective Wave A', 'direction': 'Opposite to Wave 5'}
            elif current_wave.wave_type == WaveType.CORRECTIVE_A:
                return {'type': 'Corrective Wave B', 'direction': 'Partial retracement of A'}
            elif current_wave.wave_type == WaveType.CORRECTIVE_B:
                return {'type': 'Corrective Wave C', 'direction': 'Same as Wave A'}
            elif current_wave.wave_type == WaveType.CORRECTIVE_C:
                return {'type': 'New Impulse Cycle', 'direction': 'New trend direction'}
            else:
                return None
        
        except Exception as e:
            logger.debug(f"Error predicting next wave: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.data.data_loader import DataLoader
    
    # Load sample data
    loader = DataLoader()
    data = loader.get_yahoo_data("AAPL", period="1y")
    
    # Detect waves
    detector = WaveDetector()
    waves = detector.detect_waves(data)
    
    print(f"Detected {len(waves)} Elliott Waves")
    
    # Print wave details
    for i, wave in enumerate(waves[:5]):  # Show first 5 waves
        print(f"Wave {i+1}: {wave.wave_type.value} "
              f"({wave.start_point.timestamp.strftime('%Y-%m-%d')} -> "
              f"{wave.end_point.timestamp.strftime('%Y-%m-%d')}) "
              f"Confidence: {wave.confidence:.2f}")
    
    # Get current wave count
    current_analysis = detector.get_current_wave_count(data)
    print(f"\nCurrent Analysis: {current_analysis}")
