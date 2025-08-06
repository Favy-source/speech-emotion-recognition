"""
Configuration management utilities.
"""
import yaml
from pathlib import Path
from typing import Dict, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file."""
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")
        
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise
