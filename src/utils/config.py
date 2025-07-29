"""
Configuration management for Elliott Wave Trading Bot.
"""

import yaml
import os
from typing import Any, Optional, Dict
from pathlib import Path


class ConfigManager:
    """
    Manages configuration settings for the Elliott Wave Trading Bot.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            # Look for config.yaml in project root
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            config_path = project_root / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config_data = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as file:
                    return yaml.safe_load(file)
            else:
                print(f"Warning: Config file not found at {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'yahoo_finance': {
                'enabled': True,
                'default_period': '2y',
                'default_interval': '1d'
            },
            'binance': {
                'enabled': False,
                'api_key': '',
                'api_secret': '',
                'testnet': True
            },
            'wave_detection': {
                'zigzag_threshold': 0.05,
                'min_wave_length': 5,
                'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618],
                'confidence_threshold': 0.7
            },
            'ml_models': {
                'use_ml': True,
                'model_type': 'xgboost',
                'retrain_frequency': 30,
                'feature_lookback': 50
            },
            'visualization': {
                'default_theme': 'plotly_dark',
                'show_fibonacci': True,
                'show_volume': True,
                'chart_height': 800,
                'chart_width': 1200
            },
            'backtesting': {
                'initial_capital': 100000,
                'commission': 0.001,
                'slippage': 0.0005,
                'max_positions': 5,
                'risk_per_trade': 0.02
            },
            'logging': {
                'level': 'INFO',
                'file_path': 'logs/elliott_bot.log',
                'max_file_size': '10MB',
                'backup_count': 5
            },
            'database': {
                'type': 'sqlite',
                'path': 'data/elliott_bot.db'
            },
            'risk_management': {
                'max_drawdown': 0.15,
                'var_confidence': 0.05,
                'position_sizing': 'fixed_fractional'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'yahoo_finance.enabled')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'yahoo_finance.enabled')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config_data
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save(self):
        """
        Save current configuration to file.
        """
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config_data, file, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def reload(self):
        """
        Reload configuration from file.
        """
        self.config_data = self._load_config()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Section configuration dictionary
        """
        return self.config_data.get(section, {})
    
    def update_section(self, section: str, values: Dict[str, Any]):
        """
        Update entire configuration section.
        
        Args:
            section: Section name
            values: New values for the section
        """
        if section not in self.config_data:
            self.config_data[section] = {}
        
        self.config_data[section].update(values)
    
    def validate_config(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid
        """
        try:
            # Check required sections
            required_sections = [
                'wave_detection', 'visualization', 'backtesting', 
                'logging', 'database'
            ]
            
            for section in required_sections:
                if section not in self.config_data:
                    print(f"Missing required section: {section}")
                    return False
            
            # Validate specific settings
            zigzag_threshold = self.get('wave_detection.zigzag_threshold', 0)
            if not (0 < zigzag_threshold < 1):
                print("Invalid zigzag_threshold: must be between 0 and 1")
                return False
            
            initial_capital = self.get('backtesting.initial_capital', 0)
            if initial_capital <= 0:
                print("Invalid initial_capital: must be positive")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error validating config: {e}")
            return False


# Global configuration instance
_config_instance = None

def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get global configuration instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigManager instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    return _config_instance


if __name__ == "__main__":
    # Example usage
    config = ConfigManager()
    
    print("Configuration validation:", config.validate_config())
    print("ZigZag threshold:", config.get('wave_detection.zigzag_threshold'))
    print("Chart height:", config.get('visualization.chart_height'))
    
    # Update a setting
    config.set('wave_detection.zigzag_threshold', 0.03)
    print("Updated ZigZag threshold:", config.get('wave_detection.zigzag_threshold'))
