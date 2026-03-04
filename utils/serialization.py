"""Serialization utilities"""

import json
import pickle
from typing import Any, Dict, Union
from pathlib import Path

def serialize_object(obj: Any, format: str = "json") -> Union[str, bytes]:
    """
    Serialize object
    
    Args:
        obj: Object to serialize
        format: Serialization format ("json" or "pickle")
        
    Returns:
        Serialized data
    """
    if format == "json":
        return json.dumps(obj, ensure_ascii=False, indent=2)
    elif format == "pickle":
        return pickle.dumps(obj)
    else:
        raise ValueError(f"Unsupported serialization format: {format}")

def deserialize_object(data: Union[str, bytes], format: str = "json") -> Any:
    """
    Deserialize object
    
    Args:
        data: Serialized data
        format: Serialization format
        
    Returns:
        Deserialized object
    """
    if format == "json":
        return json.loads(data)
    elif format == "pickle":
        return pickle.loads(data)
    else:
        raise ValueError(f"Unsupported deserialization format: {format}")

def save_to_file(obj: Any, filepath: Union[str, Path], format: str = "json") -> None:
    """Save object to file"""
    filepath = Path(filepath)
    data = serialize_object(obj, format)
    
    mode = "w" if format == "json" else "wb"
    with open(filepath, mode) as f:
        f.write(data)

def load_from_file(filepath: Union[str, Path], format: str = "json") -> Any:
    """Load object from file"""
    filepath = Path(filepath)
    mode = "r" if format == "json" else "rb"
    
    with open(filepath, mode) as f:
        data = f.read()
    
    return deserialize_object(data, format)