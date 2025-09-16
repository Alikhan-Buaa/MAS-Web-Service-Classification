"""
Common utility functions used across the project
Centralized helper functions for data loading, saving, logging, etc.
"""

import json
import yaml
import pickle
import joblib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import os
import sys

# Add config to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import LOGGING_CONFIG, DATA_CONFIG, RESULTS_CONFIG


def setup_logging(log_file: Optional[Path] = None, 
                 level: str = "INFO", 
                 format_str: Optional[str] = None) -> None:
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
        level: Logging level
        format_str: Log format string
    """
    if format_str is None:
        format_str = LOGGING_CONFIG['format']
    
    logging_level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = logging.Formatter(format_str)
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if LOGGING_CONFIG['handlers']['console']:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if LOGGING_CONFIG['handlers']['file'] and log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def load_data(file_path: Union[str, Path], 
              file_type: Optional[str] = None) -> Any:
    """
    Load data from various file formats
    
    Args:
        file_path: Path to the data file
        file_type: Type of file ('csv', 'json', 'yaml', 'pickle', 'joblib')
    
    Returns:
        Loaded data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Infer file type from extension if not provided
    if file_type is None:
        file_type = file_path.suffix.lower().lstrip('.')
    
    try:
        if file_type == 'csv':
            return pd.read_csv(file_path)
        elif file_type in ['json', 'jsonl']:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_type == 'jsonl':
                    return [json.loads(line) for line in f]
                return json.load(f)
        elif file_type in ['yaml', 'yml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        elif file_type in ['pickle', 'pkl']:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif file_type == 'joblib':
            return joblib.load(file_path)
        elif file_type == 'npy':
            return np.load(file_path)
        elif file_type == 'npz':
            return np.load(file_path, allow_pickle=True)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {str(e)}")
        raise


def save_data(data: Any, 
              file_path: Union[str, Path], 
              file_type: Optional[str] = None,
              **kwargs) -> None:
    """
    Save data to various file formats
    
    Args:
        data: Data to save
        file_path: Path to save the file
        file_type: Type of file ('csv', 'json', 'yaml', 'pickle', 'joblib')
        **kwargs: Additional arguments for specific save functions
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Infer file type from extension if not provided
    if file_type is None:
        file_type = file_path.suffix.lower().lstrip('.')
    
    try:
        if file_type == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=kwargs.get('index', False))
            else:
                pd.DataFrame(data).to_csv(file_path, index=kwargs.get('index', False))
        elif file_type == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=kwargs.get('indent', 2), 
                         ensure_ascii=kwargs.get('ensure_ascii', False))
        elif file_type in ['yaml', 'yml']:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, 
                         allow_unicode=kwargs.get('allow_unicode', True))
        elif file_type in ['pickle', 'pkl']:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=kwargs.get('protocol', pickle.HIGHEST_PROTOCOL))
        elif file_type == 'joblib':
            joblib.dump(data, file_path, compress=kwargs.get('compress', 3))
        elif file_type == 'npy':
            np.save(file_path, data)
        elif file_type == 'npz':
            if isinstance(data, dict):
                np.savez_compressed(file_path, **data)
            else:
                np.savez_compressed(file_path, data=data)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        logging.error(f"Error saving file {file_path}: {str(e)}")
        raise


def save_results(results: Dict[str, Any], 
                model_name: str, 
                feature_type: str, 
                n_categories: int,
                result_type: str = "evaluation") -> None:
    """
    Save model results in multiple formats
    
    Args:
        results: Results dictionary to save
        model_name: Name of the model
        feature_type: Type of features used
        n_categories: Number of categories
        result_type: Type of results ('evaluation', 'training', 'predictions')
    """
    from config import get_results_filename, get_results_dir
    
    # Determine model type
    model_type = 'dl' if model_name in ['bilstm'] else 'ml'
    
    # Get results directory
    results_dir = get_results_dir(model_type, n_categories)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    results['metadata'] = {
        'model_name': model_name,
        'feature_type': feature_type,
        'n_categories': n_categories,
        'result_type': result_type,
        'timestamp': datetime.now().isoformat(),
        'total_samples': results.get('total_samples', 'unknown')
    }
    
    # Save in multiple formats
    base_filename = f"{model_name}_{feature_type}_top{n_categories}_{result_type}"
    
    for format_type in ['json', 'yaml']:
        filename = f"{base_filename}.{format_type}"
        save_data(results, results_dir / filename, format_type)
    
    logging.info(f"Results saved for {model_name} with {feature_type} features (top {n_categories})")


def load_model(model_path: Union[str, Path]) -> Any:
    """
    Load a trained model
    
    Args:
        model_path: Path to the model file
    
    Returns:
        Loaded model
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    file_extension = model_path.suffix.lower()
    
    if file_extension == '.joblib':
        return joblib.load(model_path)
    elif file_extension in ['.pkl', '.pickle']:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    elif file_extension == '.h5':
        # For Keras/TensorFlow models
        try:
            from tensorflow.keras.models import load_model as keras_load_model
            return keras_load_model(model_path)
        except ImportError:
            raise ImportError("TensorFlow/Keras not available for loading .h5 models")
    else:
        raise ValueError(f"Unsupported model file format: {file_extension}")


def save_model(model: Any, 
               model_path: Union[str, Path], 
               metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save a trained model
    
    Args:
        model: Model to save
        model_path: Path to save the model
        metadata: Optional metadata to save alongside
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_extension = model_path.suffix.lower()
    
    try:
        if file_extension == '.joblib':
            joblib.dump(model, model_path, compress=3)
        elif file_extension in ['.pkl', '.pickle']:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif file_extension == '.h5':
            # For Keras/TensorFlow models
            model.save(model_path)
        else:
            raise ValueError(f"Unsupported model file format: {file_extension}")
        
        # Save metadata if provided
        if metadata:
            metadata_path = model_path.with_suffix('.json')
            save_data(metadata, metadata_path, 'json')
            
        logging.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logging.error(f"Error saving model to {model_path}: {str(e)}")
        raise


def create_directory_structure():
    """Create the complete directory structure for the project"""
    from config import DATA_PATH, MODELS_PATH, RESULTS_PATH, LOGS_PATH
    
    directories = [
        DATA_PATH / "raw",
        DATA_PATH / "processed",
        DATA_PATH / "splits",
        DATA_PATH / "analysis" / "plots",
        DATA_PATH / "analysis" / "statistics", 
        DATA_PATH / "analysis" / "reports",
        DATA_PATH / "features" / "tfidf",
        DATA_PATH / "features" / "sbert",
        MODELS_PATH / "saved_models" / "ml_models",
        MODELS_PATH / "saved_models" / "dl_models",
        MODELS_PATH / "model_configs" / "ml_configs",
        MODELS_PATH / "model_configs" / "dl_configs",
        RESULTS_PATH / "ml_models" / "category_wise",
        RESULTS_PATH / "dl_models" / "category_wise",
        RESULTS_PATH / "benchmarks",
        RESULTS_PATH / "topk_analysis",
        RESULTS_PATH / "overall_comparisons",
        LOGS_PATH
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    logging.info("Directory structure created successfully")


def validate_dataset(df: pd.DataFrame, 
                     text_column: str, 
                     target_column: str) -> Dict[str, Any]:
    """
    Validate dataset quality and characteristics
    
    Args:
        df: DataFrame to validate
        text_column: Name of text column
        target_column: Name of target column
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_samples': len(df),
        'missing_text': df[text_column].isnull().sum(),
        'missing_labels': df[target_column].isnull().sum(),
        'empty_text': (df[text_column].str.strip() == '').sum(),
        'unique_labels': df[target_column].nunique(),
        'label_distribution': df[target_column].value_counts().to_dict(),
        'avg_text_length': df[text_column].str.len().mean(),
        'min_text_length': df[text_column].str.len().min(),
        'max_text_length': df[text_column].str.len().max(),
        'duplicates': df.duplicated().sum()
    }
    
    # Check for issues
    issues = []
    if validation_results['missing_text'] > 0:
        issues.append(f"Missing text values: {validation_results['missing_text']}")
    if validation_results['missing_labels'] > 0:
        issues.append(f"Missing labels: {validation_results['missing_labels']}")
    if validation_results['empty_text'] > 0:
        issues.append(f"Empty text entries: {validation_results['empty_text']}")
    if validation_results['duplicates'] > 0:
        issues.append(f"Duplicate entries: {validation_results['duplicates']}")
    
    validation_results['issues'] = issues
    validation_results['is_valid'] = len(issues) == 0
    
    return validation_results


def get_timestamp() -> str:
    """Get current timestamp as formatted string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_reproducibility(seed: int = 42) -> None:
    """
    Ensure reproducibility by setting random seeds
    
    Args:
        seed: Random seed value
    """
    import random
    import os
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set environment variables for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set TensorFlow/Keras seeds if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    # Set PyTorch seeds if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    logging.info(f"Reproducibility ensured with seed: {seed}")


def format_metrics(metrics: Dict[str, float], 
                  decimal_places: int = 4) -> Dict[str, str]:
    """
    Format metrics for display
    
    Args:
        metrics: Dictionary of metric values
        decimal_places: Number of decimal places
    
    Returns:
        Dictionary of formatted metric strings
    """
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            formatted[key] = f"{value:.{decimal_places}f}"
        else:
            formatted[key] = str(value)
    return formatted


def compute_class_weights(y: Union[np.ndarray, pd.Series]) -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets
    
    Args:
        y: Target labels
    
    Returns:
        Dictionary mapping class indices to weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_classes = np.unique(y)
    class_weights = compute_class_weight(
        'balanced', 
        classes=unique_classes, 
        y=y
    )
    
    return dict(zip(unique_classes, class_weights))


def memory_usage_check() -> Dict[str, str]:
    """
    Check current memory usage
    
    Returns:
        Dictionary with memory usage information
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': f"{memory_info.rss / 1024 / 1024:.2f}",
        'vms_mb': f"{memory_info.vms / 1024 / 1024:.2f}",
        'percent': f"{process.memory_percent():.2f}",
        'available_gb': f"{psutil.virtual_memory().available / 1024 / 1024 / 1024:.2f}"
    }


def print_section_header(title: str, char: str = "=", width: int = 80) -> None:
    """
    Print a formatted section header
    
    Args:
        title: Section title
        char: Character to use for the line
        width: Total width of the header
    """
    title_len = len(title)
    if title_len >= width - 4:
        print(char * width)
        print(f"  {title}")
        print(char * width)
    else:
        padding = (width - title_len - 2) // 2
        line = char * padding + f" {title} " + char * padding
        if len(line) < width:
            line += char
        print(line)


# Export commonly used functions
__all__ = [
    'setup_logging', 'load_data', 'save_data', 'save_results',
    'load_model', 'save_model', 'create_directory_structure',
    'validate_dataset', 'get_timestamp', 'ensure_reproducibility',
    'format_metrics', 'compute_class_weights', 'memory_usage_check',
    'print_section_header'
]