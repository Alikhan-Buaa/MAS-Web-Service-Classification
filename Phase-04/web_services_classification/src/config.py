"""
Configuration file for Web Services Classification Project
Enhanced to ensure consistency across all model types
"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent

DATA_PATH = PROJECT_ROOT / "data"
MODELS_PATH = PROJECT_ROOT / "models"
RESULTS_PATH = PROJECT_ROOT / "results"
LOGS_PATH = PROJECT_ROOT / "logs"

# Random seed for reproducibility
RANDOM_SEED = 42

# Category sizes to process
CATEGORY_SIZES = [50]

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': {
        'console': True,
        'file': True
    },
    'log_files': {
        'data_analysis': LOGS_PATH / 'data_analysis.log',
        'preprocessing': LOGS_PATH / 'preprocessing.log',
        'feature_extraction': LOGS_PATH / 'feature_extraction.log',
        'training': LOGS_PATH / 'training.log',
        'evaluation': LOGS_PATH / 'evaluation.log'
    }
}

# Data configuration
DATA_CONFIG = {
    'raw_data_path': DATA_PATH / "raw" / "web_services_dataset.csv",  # Specific file path
    'processed_data_path': DATA_PATH / "processed",
    'analysis_path': DATA_PATH / "analysis",
    'text_column': 'Service Description',  # Column name for text data
    'target_column': 'Service Classification',  # Column name for categories
}

# Analysis paths (used by data_analysis.py)
ANALYSIS_PATH = DATA_CONFIG['analysis_path']

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    'splits': str(DATA_PATH / "splits" / "top_{n}_categories"),
    'labels': str(DATA_PATH / "splits" / "top_{n}_categories" / "labels.yaml"),
    'processed_data': str(DATA_PATH / "processed" / "top_{n}_categories"),
    'cleaned_data': str(DATA_PATH / "processed" / "cleaned_data.csv"),
    'features_path': DATA_PATH / "features",
    # Text preprocessing settings
    'remove_stopwords': True,
    'lemmatization': True,
    'remove_numbers': True,
    'min_word_length': 2,
    'max_word_length': 50,
    'custom_stopwords': []
}

# Split configuration (used by data_preprocessing.py)
SPLIT_CONFIG = {
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': RANDOM_SEED,
    'stratify': True
}

# Feature extraction configuration
FEATURES_CONFIG = {
    'tfidf_path': str(DATA_PATH / "features" / "tfidf" / "top_{n}_categories"),
    'sbert_path': str(DATA_PATH / "features" / "sbert" / "top_{n}_categories"),
    'plots': str(DATA_PATH / "features" / "plots"),
    'tfidf': {
        'max_features': 10000,
        'ngram_range': (1, 2),
        'stop_words': 'english'
    },
    'sbert': {
        'model_name': 'all-MiniLM-L6-v2',
        'show_progress_bar': True,
        'batch_size': 32,
        'normalize_embeddings': True
    }
}

# Preprocessing path alias (referenced in other modules)
PREPROCESS_PATH = PREPROCESSING_CONFIG['features_path']

# Feature extraction configuration alias (for backward compatibility)
FEATURE_EXTRACTION_CONFIG = FEATURES_CONFIG

# Results configuration - CRITICAL for proper file organization
RESULTS_CONFIG = {
    # ML Results
    'ml_results_path': RESULTS_PATH / "ml",
    'ml_comparisons_path': RESULTS_PATH / "ml" / "comparisons",
    'ml_category_paths': {
        n: RESULTS_PATH / "ml" / f"top_{n}_categories" for n in CATEGORY_SIZES
    },
    
    # DL Results  
    'dl_results_path': RESULTS_PATH / "dl",
    'dl_comparisons_path': RESULTS_PATH / "dl" / "comparisons",
    'dl_category_paths': {
        n: RESULTS_PATH / "dl" / f"top_{n}_categories" for n in CATEGORY_SIZES
    },
    
    # BERT Results
    'bert_results_path': RESULTS_PATH / "bert",
    'bert_comparisons_path': RESULTS_PATH / "bert" / "comparisons",
    'bert_category_paths': {
        n: RESULTS_PATH / "bert" / f"top_{n}_categories" for n in CATEGORY_SIZES
    },
    
    # DeepSeek Results
    'deepseek_results_path': RESULTS_PATH / "deepseek", 
    'deepseek_comparisons_path': RESULTS_PATH / "deepseek" / "comparisons",
    'deepseek_category_paths': {
        n: RESULTS_PATH / "deepseek" / f"top_{n}_categories" for n in CATEGORY_SIZES
    },
    
    # fusion Results
    'fusion_results_path': RESULTS_PATH / "fusion",
    'fusion_comparisons_path': RESULTS_PATH / "fusion" / "comparisons",
    'fusion_category_paths': {
        n: RESULTS_PATH / "fusion" / f"top_{n}_categories" for n in CATEGORY_SIZES
    },
    # Overall Results
    'overall_results_path': RESULTS_PATH / "overall",
}

# Saved models configuration - CRITICAL for model storage
SAVED_MODELS_CONFIG = {
    'ml_models_path': MODELS_PATH / "saved_models" / "ml_models",
    'dl_models_path': MODELS_PATH / "saved_models" / "dl_models", 
    'bert_models_path': MODELS_PATH / "saved_models" / "bert_models",
    'deepseek_models_path': MODELS_PATH / "saved_models" / "deepseek_models",
    'fusion_models_path': MODELS_PATH / "saved_models" / "fusion_models" 
}

# ML Models configuration
ML_CONFIG = {
    'models': ['LogisticRegression', 'RandomForest', 'XGBoost'],
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    },
    'random_forest': {
        'n_estimators': 100,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    },
    'xgboost': {
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'eval_metric': 'mlogloss'
    }
}

# DL Models configuration
DL_CONFIG = {
    'models': ['BiLSTM'],
    'feature_types': ['tfidf', 'sbert'],
    'bilstm': {
        'lstm_units': 128,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10,
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy']
    },
    'callbacks': {
        'early_stopping': {
            'monitor': 'val_accuracy',
            'patience': 3,
            'restore_best_weights': True
        },
        'model_checkpoint': {
            'monitor': 'val_accuracy',
            'save_best_only': True,
            'save_weights_only': False
        },
        'reduce_lr': {
            'monitor': 'val_loss',
            'factor': 0.5,
            'patience': 2,
            'min_lr': 1e-6
        }
    }
}

# BERT Models configuration
BERT_CONFIG = {
    'available_models': {
        'roberta_base': 'roberta-base',
        'roberta_large': 'roberta-large'
    },
    'model_name': 'roberta-base',  # Default model
    'max_length': 512,
    'num_train_epochs': 3,
    'eval_strategy': 'epoch',
    'logging_strategy': 'epoch',
    'logging_steps': 100,
    'save_strategy': 'epoch',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_accuracy',
    'greater_is_better': True,
    'seed': RANDOM_SEED,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_steps': 500,
    'per_device_train_batch_size': 16,  # Default batch size
    'per_device_eval_batch_size': 32,   # Default eval batch size
    'batch_sizes': {
        'roberta-base': {
            'train_batch_size': 16,
            'eval_batch_size': 32
        },
        'roberta-large': {
            'train_batch_size': 8,
            'eval_batch_size': 16
        }
    }
}

# DeepSeek Models configuration
DEEPSEEK_CONFIG = {
    'available_models': {
        'deepseek_7b_base': 'deepseek-ai/deepseek-llm-7b-base'
    },
    'model_name': 'deepseek-ai/deepseek-llm-7b-base',  # Default model
    'trust_remote_code': True,
    'max_length': 512,
    'padding': 'max_length',
    'truncation': True,
    'num_train_epochs': 3,
    'eval_strategy': 'epoch',
    'per_device_train_batch_size': 4,
    'per_device_eval_batch_size': 8,
    'gradient_accumulation_steps': 4,
    'logging_steps': 100,
    'save_strategy': 'epoch',
    'save_total_limit': 2,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_accuracy',
    'greater_is_better': True,
    'random_state': RANDOM_SEED,
    'learning_rate': 1e-4,
    'gradient_checkpointing': True,
    'quantization': {
        'load_in_4bit': True,
        'bnb_4bit_use_double_quant': True,
        'bnb_4bit_quant_type': 'nf4',
        'bnb_4bit_compute_dtype': 'float16'
    },
    'lora': {
        'task_type': 'SEQ_CLS',
        'r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'bias': 'none'
    },
    'text_preprocessing': {
        'clean_text': True
    },
    'batch_sizes': {
        'deepseek-ai/deepseek-llm-7b-base': {
            'train_batch_size': 4,
            'eval_batch_size': 8
        }
    }
}

# ============================================================================
# FUSION CONFIGURATION (Single Hybrid: RoBERTa + DeepSeek)
# ============================================================================

FUSION_CONFIG = {
    'available_models': {
        'deepseek_base': 'deepseek-ai/deepseek-llm-7b-base',
        'roberta_base': 'roberta-base'
    },
    'deepseek_model': 'deepseek-ai/deepseek-llm-7b-base',  # Fixed DeepSeek model
    'roberta_model': 'roberta-base',  # Fixed RoBERTa model
    'fusion_types': ['concat', 'average', 'weighted', 'gating'],  # Different fusion strategies
    'common_dim': 768,  # Common embedding dimension
    'max_length': 128,
    'num_train_epochs': 10,
    'batch_size': 8,
    'eval_batch_size': 16,
    'learning_rate': 1e-5,
    'weight_decay': 0.01,
    'gradient_clip': 1.0,
    'dropout': 0.3,
    'scheduler': {
        'mode': 'max',
        'patience': 2,
        'factor': 0.5
    }
}

def create_all_directories():
    """Create all necessary directories"""
    directories = [
        DATA_PATH / "raw",
        DATA_PATH / "processed", 
        DATA_PATH / "splits",
        DATA_PATH / "features" / "tfidf",
        DATA_PATH / "features" / "sbert",
        DATA_PATH / "features" / "plots",
        DATA_PATH / "analysis",
        MODELS_PATH / "saved_models" / "ml_models",
        MODELS_PATH / "saved_models" / "dl_models",
        MODELS_PATH / "saved_models" / "bert_models", 
        MODELS_PATH / "saved_models" / "deepseek_models",
        MODELS_PATH / "saved_models" / "fusion_models",
        RESULTS_PATH / "ml" / "comparisons",
        RESULTS_PATH / "dl" / "comparisons",
        RESULTS_PATH / "bert" / "comparisons",
        RESULTS_PATH / "deepseek" / "comparisons",
        RESULTS_PATH / "fusion" / "comparisons",
        RESULTS_PATH / "overall",
        LOGS_PATH
    ]
    
    # Create category-specific directories
    for n_categories in CATEGORY_SIZES:
        directories.extend([
            DATA_PATH / "splits" / f"top_{n_categories}_categories",
            DATA_PATH / "processed" / f"top_{n_categories}_categories",
            DATA_PATH / "features" / "tfidf" / f"top_{n_categories}_categories",
            DATA_PATH / "features" / "sbert" / f"top_{n_categories}_categories",
            DATA_PATH / "analysis" / f"top_{n_categories}_categories",
            RESULTS_PATH / "ml" / f"top_{n_categories}_categories",
            RESULTS_PATH / "dl" / f"top_{n_categories}_categories", 
            RESULTS_PATH / "bert" / f"top_{n_categories}_categories",
            RESULTS_PATH / "deepseek" / f"top_{n_categories}_categories",
            RESULTS_PATH / "fusion" / f"top_{n_categories}_categories"
        ])
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print(f"Created {len(directories)} directories")

if __name__ == "__main__":
    create_all_directories()
    print("Configuration initialized and directories created.")
