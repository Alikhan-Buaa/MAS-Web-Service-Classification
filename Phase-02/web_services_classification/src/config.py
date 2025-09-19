"""
Configuration settings for Web Service Classification Project
Updated to include BERT model configuration

Centralizes all configuration settings used across the project.
Ensures consistency, maintainability, and easy management of paths,
hyperparameters, and evaluation options.

Project Structure:
- src/               → source code
- data/              → raw, processed, and feature data
- models/            → saved ML & DL models
- results/           → analysis outputs and evaluation results
- logs/              → logging outputs
"""

from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent  # Project root (1 level up from src/)
DATA_PATH = PROJECT_ROOT / "data"
MODELS_PATH = PROJECT_ROOT / "models"
RESULTS_PATH = PROJECT_ROOT / "results"
LOGS_PATH = PROJECT_ROOT / "logs"

# =============================================================================
# General Settings
# =============================================================================
CATEGORY_SIZES = [50]  # Number of top categories to use
RANDOM_SEED = 42                       # Global seed for reproducibility

# =============================================================================
# Data Configuration
# =============================================================================
DATA_CONFIG = {
    # Input data
    "raw_data_path": DATA_PATH / "raw" / "web_services_dataset.csv",
    
    # Processed data path (added for DL models)
    "processed_data_path": DATA_PATH / "processed",

    # Column names (adjust if CSV schema changes)
    "text_column": "Service Description",          # service description text
    "target_column": "Service Classification",     # classification labels
}

# =============================================================================
# Step 1: Analysis - Uniform top_{n}_categories naming
# =============================================================================
ANALYSIS_PATH = DATA_PATH / "analysis"
ANALYSIS_CONFIG = {
    "overall": ANALYSIS_PATH / "overall",                    # global dataset stats & plots
    "category_wise": ANALYSIS_PATH / "top_{n}_categories",   # Top-N stats & distributions (uniform naming)
    "comparisons": ANALYSIS_PATH / "comparisons"             # cross-TopN comparisons
}

# =============================================================================
# Step 2: Preprocessing - Uniform top_{n}_categories naming
# =============================================================================
PREPROCESS_PATH = DATA_PATH / "processed"
PREPROCESSING_CONFIG = {
    "processed_data": str(PREPROCESS_PATH / "top_{n}_categories"),        # cleaned datasets (uniform naming)
    "splits": str(PREPROCESS_PATH / "splits" / "top_{n}_categories"),     # train/val/test splits (uniform naming)
    "labels": str(PREPROCESS_PATH / "labels_top_{n}_categories.yaml"),    # label mappings (uniform naming)

    # Basic text cleaning
    "remove_stopwords": False,
    "remove_numbers": True,
    "lemmatization": True,
    "lowercase": True,
    "remove_punctuation": True,

    # Word filtering
    "min_word_length": 2,
    "max_word_length": 50,

    # Custom stopwords
    "custom_stopwords": ["a", "an", "the", "and", "or", "but", "in", "on", "at", "to"],

    # Advanced cleaning
    "remove_urls": True,
    "remove_emails": True,
    "normalize_whitespace": True
}

# =============================================================================
# Step 3: Features - Uniform top_{n}_categories naming
# =============================================================================
FEATURES_PATH = DATA_PATH / "features"
FEATURES_CONFIG = {
    "tfidf_path": str(FEATURES_PATH / "tfidf" / "top_{n}_categories"),   # TF-IDF vectors (consistent naming)
    "sbert_path": str(FEATURES_PATH / "sbert" / "top_{n}_categories"),   # SBERT embeddings (consistent naming)
    "plots": FEATURES_PATH / "feature_plots",                            # tfidf_top_terms, sbert_clusters
    "stats": FEATURES_PATH / "feature_stats",                            # vocab/embedding stats

    # TF-IDF settings
    "tfidf": {
        "max_features": 10000,
        "ngram_range": (1, 3),
        "min_df": 2,
        "max_df": 0.95,
        "use_idf": True,
        "smooth_idf": True,
        "sublinear_tf": True
    },

    # SBERT settings
    "sbert": {
        "model_name": "all-MiniLM-L6-v2",
        "max_length": 512,
        "batch_size": 32,
        "device": "cpu",  # change to "cuda" for GPU
        "normalize_embeddings": True,
        "show_progress_bar": True,
        "convert_to_tensor": False,
        "convert_to_numpy": True
    }
}

# =============================================================================
# Data Split Configuration
# =============================================================================
SPLIT_CONFIG = {
    "train_size": 0.80,       # proportion of data for training
    "val_size": 0.10,         # proportion of data for validation
    "test_size": 0.10,        # proportion of data for testing
    "random_state": RANDOM_SEED,  # reproducibility
    "stratify": True,         # preserve label distribution in splits
    "shuffle": True           # shuffle before splitting
}

# =============================================================================
# Machine Learning Configuration
# =============================================================================
ML_CONFIG = {
    # Models to train
    "models": ["logistic_regression", "random_forest", "xgboost"],

    # Feature types to use
    "feature_types": ["tfidf", "sbert"],

    # Cross-validation settings
    "cv_folds": 5,
    "cv_scoring": "f1_macro",
    "cv_shuffle": True,
    "cv_random_state": RANDOM_SEED,

    # Model parameters
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "solver": "liblinear",
        "multi_class": "ovr"
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0,
        "reg_lambda": 1
    }
}

# =============================================================================
# Deep Learning Configuration
# =============================================================================
DL_CONFIG = {
    # Model architectures to train
    "models": ["bilstm"],
    
    # Feature types to use
    "feature_types": ["tfidf", "sbert"],
    
    # BiLSTM model parameters
    "bilstm": {
        "lstm_units": 128,
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "patience": 10,
        "validation_split": 0.2,
        "optimizer": "adam",
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"]
    },
    
    # Training callbacks
    "callbacks": {
        "early_stopping": {
            "monitor": "val_accuracy",
            "patience": 10,
            "restore_best_weights": True
        },
        "reduce_lr": {
            "monitor": "val_loss",
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-7
        },
        "model_checkpoint": {
            "monitor": "val_accuracy",
            "save_best_only": True,
            "save_weights_only": False
        }
    }
}

# =============================================================================
# BERT Configuration
# =============================================================================
BERT_CONFIG = {
    # Available RoBERTa models
    "available_models": {
        "roberta_base": "roberta-base",
        "roberta_large": "roberta-large"
    },
    
    # Default model configuration
    "model_name": "roberta-base",  # Default to base model
    "max_length": 128,
    "num_train_epochs": 10,  # Changed from 5 to 10
    
    # Training configuration  
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    
    # Model-specific batch sizes (adjust for memory)
    "batch_sizes": {
        "roberta-base": {
            "train_batch_size": 8,
            "eval_batch_size": 8
        },
        "roberta-large": {
            "train_batch_size": 4,  # Smaller batch for large model
            "eval_batch_size": 4
        }
    },
    
    # Data configuration
    "test_size": 0.2,
    "seed": RANDOM_SEED,
    
    # Logging and output configuration
    "logging_steps": 100,
    "eval_strategy": "epoch",
    "logging_strategy": "steps",
    "save_strategy": "epoch", 
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "greater_is_better": True,
    
    # Additional BERT-specific settings
    "gradient_accumulation_steps": 1,
    "fp16": False,  # Set to True if using compatible GPU
    "dataloader_drop_last": False,
    "dataloader_num_workers": 0,
    "remove_unused_columns": True,
    "report_to": [],
}

# =============================================================================
# ML Models & Results Storage - Uniform top_{n}_categories naming
# =============================================================================
SAVED_MODELS_CONFIG = {
    "ml_models_path": MODELS_PATH / "ml",  # trained ML models
    "dl_models_path": MODELS_PATH / "dl",  # trained DL models
    "bert_models_path": MODELS_PATH / "bert",  # trained BERT models
}

RESULTS_CONFIG = {
    # ML Results
    "ml_results_path": RESULTS_PATH / "ml",           # per-model results
    "ml_comparisons_path": RESULTS_PATH / "ml_comparisons",  # cross-model comparisons
    
    # DL Results
    "dl_results_path": RESULTS_PATH / "dl",           # per-model DL results
    "dl_comparisons_path": RESULTS_PATH / "dl_comparisons",  # cross-model DL comparisons
    
    # BERT Results
    "bert_results_path": RESULTS_PATH / "bert",       # per-model BERT results
    "bert_comparisons_path": RESULTS_PATH / "bert_comparisons",  # cross-model BERT comparisons

    # File naming patterns - uniform top_{n}_categories naming
    "model_results_pattern": "{model_name}_{feature_type}_top_{n}_categories",
 
    # Category-wise results directories - uniform top_{n}_categories naming
    "ml_category_paths": { 
        50: RESULTS_PATH / "ml" / "top_50_categories"
    },
    
    # DL Category-wise results directories (same structure as ML)
    "dl_category_paths": {
        50: RESULTS_PATH / "dl" / "top_50_categories"
    },
    
    # BERT Category-wise results directories
    "bert_category_paths": {
        50: RESULTS_PATH / "bert" / "top_50_categories"
    }
}

# =============================================================================
# Ensure Required Directories - Updated with uniform naming and BERT
# =============================================================================
for path in [
    ANALYSIS_PATH,
    PREPROCESS_PATH,
    FEATURES_PATH,
    MODELS_PATH,
    RESULTS_PATH,
    LOGS_PATH,
    RESULTS_PATH / "ml",
    RESULTS_PATH / "ml_comparisons",
    RESULTS_PATH / "dl",
    RESULTS_PATH / "dl_comparisons",
    RESULTS_PATH / "bert",
    RESULTS_PATH / "bert_comparisons",
    MODELS_PATH / "dl",
    MODELS_PATH / "bert",
]:
    path.mkdir(parents=True, exist_ok=True)

# Create category-specific directories with uniform naming
for n in CATEGORY_SIZES:
    category_paths = [
        ANALYSIS_PATH / f"top_{n}_categories",
        PREPROCESS_PATH / f"top_{n}_categories",
        PREPROCESS_PATH / "splits" / f"top_{n}_categories",
        FEATURES_PATH / "tfidf" / f"top_{n}_categories",
        FEATURES_PATH / "sbert" / f"top_{n}_categories",
        MODELS_PATH / "ml" / f"top_{n}_categories",
        MODELS_PATH / "dl" / f"top_{n}_categories",
        MODELS_PATH / "bert" / f"top_{n}_categories",
        RESULTS_PATH / "ml" / f"top_{n}_categories",
        RESULTS_PATH / "dl" / f"top_{n}_categories",
        RESULTS_PATH / "bert" / f"top_{n}_categories"
    ]
    for path in category_paths:
        path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Logging Configuration
# =============================================================================
LOGGING_CONFIG = {
    "level": "INFO",   # Can be "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": {
        "file": True,      # Enable logging to file
        "console": True    # Enable logging to console
    },
    "log_files": {
        "data_analysis": LOGS_PATH / "data_analysis.log",
        "preprocessing": LOGS_PATH / "preprocessing.log",
        "feature_extraction": LOGS_PATH / "feature_extraction.log",
        "ml_training": LOGS_PATH / "ml_training.log",
        "dl_training": LOGS_PATH / "dl_training.log",
        "bert_training": LOGS_PATH / "bert_training.log",
        "topk_evaluation": LOGS_PATH / "topk_evaluation.log",
        "evaluation": LOGS_PATH / "evaluation.log",
        "visualization": LOGS_PATH / "visualization.log"
    },
    "max_file_size": 10 * 1024 * 1024,  # 10 MB
    "backup_count": 5                   # Keep up to 5 rotated logs
}