"""Helper utility functions"""

import importlib
from datetime import datetime
from typing import Any, Dict, Optional, Type
from pathlib import Path

def format_time(timestamp: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format time
    
    Args:
        timestamp: Timestamp, defaults to current time
        format_str: Format string
        
    Returns:
        Formatted time string
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime(format_str)

def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate if configuration contains required keys
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Returns:
        Whether validation passed
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Configuration missing required keys: {missing_keys}")
    return True

def safe_import(module_name: str, class_name: Optional[str] = None) -> Any:
    """
    Safely import module or class
    
    Args:
        module_name: Module name
        class_name: Class name (optional)
        
    Returns:
        Imported module or class
    """
    try:
        module = importlib.import_module(module_name)
        if class_name:
            return getattr(module, class_name)
        return module
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot import {module_name}.{class_name or ''}: {e}")

def ensure_dir(path: Path) -> Path:
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent.parent

def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result