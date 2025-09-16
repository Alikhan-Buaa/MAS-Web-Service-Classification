"""
Enhanced Deep Learning Models for Web Service Classification
Includes comprehensive visualization, analysis, and result storage
"""

import pandas as pd
import numpy as np
import logging
import json
import yaml
import time
import traceback
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding, Input, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Import configuration
from src.config import (
    ML_CONFIG, DATA_CONFIG, CATEGORY_SIZES, RESULTS_CONFIG, 
    SAVED_MODELS_CONFIG, DL_CONFIG, RESULTS_PATH, PREPROCESSING_CONFIG, RANDOM_SEED
)
from src.preprocessing.feature_extraction import FeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DLModelTrainer:
    """Main class for training deep learning models with comprehensive analysis"""
    
    @staticmethod
    def make_json_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: DLModelTrainer.make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [DLModelTrainer.make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        else:
            return obj

    def __init__(self):
        self.models = {}
        self.feature_extractor = FeatureExtractor()
        self.config = DL_CONFIG['bilstm']
        self.callbacks_config = DL_CONFIG['callbacks']
        # Add: Initialize storage for final results (like ML models)
        self.dl_final_results = {}
        
        # Configure GPU memory growth
        self._configure_gpu()
        
        # Create results directories
        self._create_directories()
        
    def _configure_gpu(self):
        """Configure GPU memory growth to prevent OOM errors"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Configured GPU memory growth for {len(gpus)} GPU(s)")
            else:
                logger.info("No GPUs found, using CPU")
        except RuntimeError as e:
            logger.warning(f"GPU configuration error: {e}")
    
    def _create_directories(self):
        """Create necessary directories for results and visualizations"""
        # Create main DL directories using config paths
        directories = [
            RESULTS_CONFIG['dl_results_path'],
            RESULTS_CONFIG['dl_comparisons_path'],
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create category-specific directories using config paths
        for n_categories in CATEGORY_SIZES:
            category_dir = RESULTS_CONFIG['dl_category_paths'][n_categories]
            category_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Created result directories for DL models")
    
    def create_bilstm_model_tfidf(self, input_dim, n_classes):
        """Create Dense Neural Network for TF-IDF features"""
        try:
            logger.info(f"Creating TF-IDF model with input_dim={input_dim}, n_classes={n_classes}")
            
            model = Sequential([
                Input(shape=(input_dim,)),
                Dense(512, activation='relu', name='dense_1'),
                Dropout(self.config['dropout_rate'], name='dropout_1'),
                Dense(256, activation='relu', name='dense_2'), 
                Dropout(self.config['dropout_rate'], name='dropout_2'),
                Dense(128, activation='relu', name='dense_3'),
                Dropout(self.config['dropout_rate'], name='dropout_3'),
                Dense(n_classes, activation='softmax', name='output')
            ])
            
            optimizer = Adam(learning_rate=self.config['learning_rate'])
            model.compile(
                optimizer=optimizer,
                loss=self.config['loss'],
                metrics=self.config['metrics']
            )
            
            logger.info("TF-IDF model created successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error creating TF-IDF model: {str(e)}")
            raise
    
    def create_bilstm_model_sbert(self, input_dim, n_classes):
        """Create model for SBERT embeddings"""
        try:
            logger.info(f"Creating SBERT model with input_dim={input_dim}, n_classes={n_classes}")
            
            if input_dim >= 384:  # Typical SBERT dimension
                model = Sequential([
                    Input(shape=(input_dim,)),
                    Dense(256, activation='relu', name='dense_1'),
                    Dropout(self.config['dropout_rate'], name='dropout_1'),
                    Dense(128, activation='relu', name='dense_2'),
                    Dropout(self.config['dropout_rate'], name='dropout_2'),
                    Dense(64, activation='relu', name='dense_3'),
                    Dropout(self.config['dropout_rate'], name='dropout_3'),
                    Dense(n_classes, activation='softmax', name='output')
                ])
            else:
                model = Sequential([
                    Input(shape=(input_dim,)),
                    Dense(self.config['lstm_units'], activation='relu', name='dense_1'),
                    Dropout(self.config['dropout_rate'], name='dropout_1'),
                    Dense(self.config['lstm_units']//2, activation='relu', name='dense_2'),
                    Dropout(self.config['dropout_rate'], name='dropout_2'),
                    Dense(n_classes, activation='softmax', name='output')
                ])
            
            optimizer = Adam(learning_rate=self.config['learning_rate'])
            model.compile(
                optimizer=optimizer,
                loss=self.config['loss'],
                metrics=self.config['metrics']
            )
            
            logger.info("SBERT model created successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error creating SBERT model: {str(e)}")
            raise
    
    def prepare_data_for_dl(self, X_train, X_val, X_test, y_train, y_val, y_test, n_classes):
        """Prepare data for deep learning models"""
        try:
            logger.info("Preparing data for deep learning...")
            
            # Convert sparse matrices to dense for TF-IDF
            if hasattr(X_train, 'toarray'):
                logger.info("Converting sparse matrices to dense arrays")
                X_train = X_train.toarray()
                X_val = X_val.toarray()
                X_test = X_test.toarray()
            
            # Convert to float32 for better performance
            X_train = X_train.astype(np.float32)
            X_val = X_val.astype(np.float32)
            X_test = X_test.astype(np.float32)
            
            logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            logger.info(f"Label ranges - Train: {y_train.min()}-{y_train.max()}, Val: {y_val.min()}-{y_val.max()}")
            
            # Validate labels
            max_label = max(y_train.max(), y_val.max(), y_test.max())
            if max_label >= n_classes:
                raise ValueError(f"Label values exceed categories. Max: {max_label}, n_categories: {n_classes}")
            
            # One-hot encode labels
            y_train_encoded = to_categorical(y_train, num_classes=n_classes)
            y_val_encoded = to_categorical(y_val, num_classes=n_classes)
            y_test_encoded = to_categorical(y_test, num_classes=n_classes)
            
            logger.info(f"Encoded labels - Train: {y_train_encoded.shape}, Val: {y_val_encoded.shape}, Test: {y_test_encoded.shape}")
            
            return X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def plot_training_history(self, history, model_name, n_categories, feature_type):
        """Create training history plots for accuracy and loss"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot accuracy
            ax1.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
            ax1.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
            ax1.set_title(f'{model_name} - Training & Validation Accuracy\n{n_categories} Categories ({feature_type.upper()})')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot loss
            ax2.plot(history['loss'], label='Training Loss', linewidth=2)
            ax2.plot(history['val_loss'], label='Validation Loss', linewidth=2)
            ax2.set_title(f'{model_name} - Training & Validation Loss\n{n_categories} Categories ({feature_type.upper()})')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot using category-specific path (same as ML structure)
            plot_dir = RESULTS_CONFIG['dl_category_paths'][n_categories]
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_file = plot_dir / f'{model_name}_{feature_type}_top{n_categories}_history.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training history plot saved: {plot_file}")
            return str(plot_file)
            
        except Exception as e:
            logger.error(f"Error creating training history plot: {str(e)}")
            return None
    
    def plot_confusion_matrix(self, cm, class_labels, model_name, n_categories, feature_type):
        """Create and save confusion matrix visualization"""
        try:
            plt.figure(figsize=(max(10, n_categories), max(8, n_categories)))
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_labels, yticklabels=class_labels,
                       cbar_kws={'label': 'Count'})
            
            plt.title(f'{model_name} - Confusion Matrix\n{n_categories} Categories ({feature_type.upper()})')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save plot using category-specific path (same as ML structure)
            cm_dir = RESULTS_CONFIG['dl_category_paths'][n_categories]
            cm_dir.mkdir(parents=True, exist_ok=True)
            cm_plot_file = cm_dir / f'{model_name}_{feature_type}_top{n_categories}_confusion_matrix.png'
            plt.savefig(cm_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix plot saved: {cm_plot_file}")
            return str(cm_plot_file)
            
        except Exception as e:
            logger.error(f"Error creating confusion matrix plot: {str(e)}")
            return None
    
    def save_confusion_matrix_csv(self, cm, class_labels, model_name, n_categories, feature_type):
        """Save confusion matrix as CSV"""
        try:
            # Create DataFrame
            cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
            
            # Save CSV using category-specific path (same as ML structure)
            cm_dir = RESULTS_CONFIG['dl_category_paths'][n_categories]
            cm_dir.mkdir(parents=True, exist_ok=True)
            cm_csv_file = cm_dir / f'{model_name}_{feature_type}_top{n_categories}_confusion_matrix.csv'
            cm_df.to_csv(cm_csv_file)
            
            logger.info(f"Confusion matrix CSV saved: {cm_csv_file}")
            return str(cm_csv_file)
            
        except Exception as e:
            logger.error(f"Error saving confusion matrix CSV: {str(e)}")
            return None
    
    def calculate_top_k_accuracy(self, y_true, y_proba, k=5):
        """Calculate Top-K accuracy"""
        try:
            if k == 1:
                y_pred = np.argmax(y_proba, axis=1)
                y_true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
                return accuracy_score(y_true_labels, y_pred)
            
            if y_true.ndim > 1:
                y_true_labels = np.argmax(y_true, axis=1)
            else:
                y_true_labels = y_true
            
            top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
            correct = sum(1 for i, true_label in enumerate(y_true_labels) 
                         if true_label in top_k_preds[i])
            
            return correct / len(y_true_labels)
            
        except Exception as e:
            logger.error(f"Error calculating top-k accuracy: {str(e)}")
            return 0.0
    
    def evaluate_dl_model(self, model, X_test, y_test, model_name, n_categories, feature_type):
        """Comprehensive evaluation of deep learning model"""
        try:
            logger.info(f"Evaluating model: {model_name}")
            
            # Get predictions and probabilities
            start_time = time.time()
            y_proba = model.predict(X_test, verbose=0)
            inference_time = time.time() - start_time
            
            y_pred = np.argmax(y_proba, axis=1)
            y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
            
            macro_precision = np.mean(precision)
            macro_recall = np.mean(recall)
            macro_f1 = np.mean(f1)
            
            micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='micro', zero_division=0
            )
            
            # Top-K accuracies
            top1_accuracy = self.calculate_top_k_accuracy(y_test, y_proba, k=1)
            top3_accuracy = self.calculate_top_k_accuracy(y_test, y_proba, k=3)
            top5_accuracy = self.calculate_top_k_accuracy(y_test, y_proba, k=5)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Class labels (using category indices for now)
            class_labels = [f'Cat_{i}' for i in range(n_categories)]
            
            # Create visualizations
            cm_plot_path = self.plot_confusion_matrix(cm, class_labels, model_name, n_categories, feature_type)
            cm_csv_path = self.save_confusion_matrix_csv(cm, class_labels, model_name, n_categories, feature_type)
            
            # Classification report
            class_report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
            
            # Per-class metrics
            per_class_metrics = []
            for i in range(len(precision)):
                per_class_metrics.append({
                    'class_id': int(i),
                    'class_name': class_labels[i],
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i]),
                    'top1_accuracy': float(np.sum((y_true == i) & (y_pred == i)) / np.sum(y_true == i) if np.sum(y_true == i) > 0 else 0)
                })
            
            # Compile results
            results = {
                'model_name': model_name,
                'feature_type': feature_type,
                'n_categories': int(n_categories),
                'top1_accuracy': float(top1_accuracy),
                'top3_accuracy': float(top3_accuracy),
                'top5_accuracy': float(top5_accuracy),
                'accuracy': float(accuracy),
                'macro_precision': float(macro_precision),
                'macro_recall': float(macro_recall),
                'macro_f1': float(macro_f1),
                'micro_precision': float(micro_precision),
                'micro_recall': float(micro_recall),
                'micro_f1': float(micro_f1),
                'confusion_matrix': [[int(val) for val in row] for row in cm.tolist()],
                'confusion_matrix_plot': cm_plot_path,
                'confusion_matrix_csv': cm_csv_path,
                'per_class_metrics': per_class_metrics,
                'classification_report': class_report,
                'inference_time': float(inference_time),
                'predictions': [int(pred) for pred in y_pred.tolist()],
                'probabilities': [[float(prob) for prob in row] for row in y_proba.tolist()]
            }
            
            logger.info(f"{model_name} Evaluation Results:")
            logger.info(f"  Top-1 Accuracy: {top1_accuracy:.4f}")
            logger.info(f"  Top-3 Accuracy: {top3_accuracy:.4f}")
            logger.info(f"  Top-5 Accuracy: {top5_accuracy:.4f}")
            logger.info(f"  Macro F1: {macro_f1:.4f}")
            logger.info(f"  Micro F1: {micro_f1:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {str(e)}")
            raise

    def save_model_performance_data(self, results, model_name, n_categories, feature_type):
        """Save model performance data in the format needed for plotting"""
        # Create standardized model entry (similar to ML models)
        model_entry = {
            "model": f"{model_name}-{feature_type}",
            "accuracy": results["accuracy"],
            "precision": results["macro_precision"],
            "recall": results["macro_recall"],
            "f1_score": results["macro_f1"],
            "feature_type": feature_type,
            "n_categories": n_categories,
            "top1_accuracy": results["top1_accuracy"],
            "top3_accuracy": results["top3_accuracy"],
            "top5_accuracy": results["top5_accuracy"],
            "training_time": results.get("training_time", 0),
            "inference_time": results.get("inference_time", 0)
        }
        
        # Store in dl_final_results structure
        if n_categories not in self.dl_final_results:
            self.dl_final_results[n_categories] = []
        
        self.dl_final_results[n_categories].append(model_entry)
        
        # Save after each model
        self.save_final_results()
        
        return model_entry

    def save_final_results(self):
        """Save final results to pickle file for plotting"""
        results_dir = SAVED_MODELS_CONFIG["dl_models_path"]
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "dl_final_results.pkl"
        with open(results_file, "wb") as f:
            pickle.dump(self.dl_final_results, f)
        logger.info(f"DL final results saved to {results_file}")

        results_dir = RESULTS_CONFIG["dl_comparisons_path"]
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "dl_final_results.pkl"
        with open(results_file, "wb") as f:
            pickle.dump(self.dl_final_results, f)
        logger.info(f"DL final results saved to {results_file}")

    def print_model_metrics(self, results, model_name, n_categories, feature_type, training_time):
        """Print model metrics to console in a formatted way"""
        print(f"\n{'='*60}")
        print(f"DL MODEL PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Feature Type: {feature_type.upper()}")
        print(f"Categories: Top {n_categories}")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Inference Time: {results['inference_time']:.4f} seconds")
        print(f"{'-'*60}")
        print(f"ACCURACY METRICS:")
        print(f"  Accuracy:      {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"  Top-1 Acc:     {results['top1_accuracy']:.4f} ({results['top1_accuracy']*100:.2f}%)")
        print(f"  Top-3 Acc:     {results['top3_accuracy']:.4f} ({results['top3_accuracy']*100:.2f}%)")
        print(f"  Top-5 Acc:     {results['top5_accuracy']:.4f} ({results['top5_accuracy']*100:.2f}%)")
        print(f"{'-'*60}")
        print(f"CLASSIFICATION METRICS (Macro Average):")
        print(f"  Precision:     {results['macro_precision']:.4f} ({results['macro_precision']*100:.2f}%)")
        print(f"  Recall:        {results['macro_recall']:.4f} ({results['macro_recall']*100:.2f}%)")
        print(f"  F1-Score:      {results['macro_f1']:.4f} ({results['macro_f1']*100:.2f}%)")
        print(f"{'='*60}\n")
    
    def train_model_on_category(self, n_categories, feature_type='sbert'):
        """Train model on a specific category size"""
        try:
            logger.info(f"Training model for {n_categories} categories with {feature_type} features...")
            
            tf.keras.backend.clear_session()
            
            # Load datasets using correct config paths
            splits_dir = Path(PREPROCESSING_CONFIG["splits"].format(n=n_categories))
            if not splits_dir.exists():
                raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
            
            train_df = pd.read_csv(splits_dir / 'train.csv')
            val_df = pd.read_csv(splits_dir / 'val.csv')
            test_df = pd.read_csv(splits_dir / 'test.csv')
            
            logger.info(f"Loaded datasets - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            
            # Get features
            if feature_type == 'tfidf':
                logger.info("Loading TF-IDF features...")
                self.feature_extractor.load_tfidf_vectorizer(n_categories)
                X_train = self.feature_extractor.tfidf_vectorizer.transform(train_df['cleaned_text'])
                X_val = self.feature_extractor.tfidf_vectorizer.transform(val_df['cleaned_text'])
                X_test = self.feature_extractor.tfidf_vectorizer.transform(test_df['cleaned_text'])
            
            elif feature_type == 'sbert':
                logger.info("Loading SBERT features...")
                X_train = self.feature_extractor.load_sbert_features(n_categories, 'train')
                X_val = self.feature_extractor.load_sbert_features(n_categories, 'val')
                X_test = self.feature_extractor.load_sbert_features(n_categories, 'test')
            
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")
            
            # Get labels
            y_train = train_df['encoded_label'].values
            y_val = val_df['encoded_label'].values
            y_test = test_df['encoded_label'].values
            
            # Prepare data
            X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded = self.prepare_data_for_dl(
                X_train, X_val, X_test, y_train, y_val, y_test, n_categories
            )
            
            # Create model
            input_dim = X_train.shape[1]
            logger.info(f"Model input dimension: {input_dim}, classes: {n_categories}")
            
            if feature_type == 'tfidf':
                model = self.create_bilstm_model_tfidf(input_dim, n_categories)
            else:
                model = self.create_bilstm_model_sbert(input_dim, n_categories)
            
            # Setup callbacks and paths
            model_dir = SAVED_MODELS_CONFIG['dl_models_path'] / f'top_{n_categories}_categories'
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_filename = f'BiLSTM_{feature_type}_top{n_categories}_model.h5'
            model_path = model_dir / model_filename
            
            callbacks = [
                EarlyStopping(
                    monitor=self.callbacks_config['early_stopping']['monitor'],
                    patience=self.callbacks_config['early_stopping']['patience'],
                    restore_best_weights=self.callbacks_config['early_stopping']['restore_best_weights'],
                    verbose=1
                ),
                ModelCheckpoint(
                    str(model_path),
                    monitor=self.callbacks_config['model_checkpoint']['monitor'],
                    save_best_only=self.callbacks_config['model_checkpoint']['save_best_only'],
                    save_weights_only=self.callbacks_config['model_checkpoint']['save_weights_only'],
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor=self.callbacks_config['reduce_lr']['monitor'],
                    factor=self.callbacks_config['reduce_lr']['factor'],
                    patience=self.callbacks_config['reduce_lr']['patience'],
                    min_lr=self.callbacks_config['reduce_lr']['min_lr'],
                    verbose=1
                )
            ]
            
            # Training parameters from config
            batch_size = min(self.config['batch_size'], len(X_train) // 10, 32)
            epochs = self.config['epochs']
            
            logger.info(f"Training with batch_size={batch_size}, epochs={epochs}")
            
            # Train model
            start_time = time.time()
            history = model.fit(
                X_train, y_train_encoded,
                validation_data=(X_val, y_val_encoded),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Load best model
            if model_path.exists():
                model.load_weights(str(model_path))
                logger.info("Loaded best model weights")
            
            # Create training history plot
            history_dict = {
                'loss': [float(x) for x in history.history['loss']],
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']]
            }
            
            model_name = f"BiLSTM_{feature_type}"
            history_plot_path = self.plot_training_history(history_dict, model_name, n_categories, feature_type)
            
            # Save training history as JSON and YAML
            history_dir = RESULTS_CONFIG['dl_category_paths'][n_categories]
            
            # JSON format
            history_json_file = history_dir / f'{model_name}_{feature_type}_top{n_categories}_history.json'
            with open(history_json_file, 'w') as f:
                json.dump(history_dict, f, indent=2)
            
            # YAML format
            history_yaml_file = history_dir / f'{model_name}_{feature_type}_top{n_categories}_history.yaml'
            with open(history_yaml_file, 'w') as f:
                yaml.dump(history_dict, f, default_flow_style=False)
            
            # Evaluate model
            eval_results = self.evaluate_dl_model(model, X_test, y_test_encoded, model_name, n_categories, feature_type)
            eval_results['training_time'] = float(training_time)
            eval_results['training_history'] = history_dict
            eval_results['training_history_plot'] = history_plot_path
            eval_results['training_history_json'] = str(history_json_file)
            eval_results['training_history_yaml'] = str(history_yaml_file)
            
            # Print metrics to console
            self.print_model_metrics(eval_results, model_name, n_categories, feature_type, training_time)

            # Save performance data for plotting
            self.save_model_performance_data(eval_results, model_name, n_categories, feature_type)
            
            logger.info(f"Model saved to {model_path}")
            
            return eval_results
            
        except Exception as e:
            logger.error(f"Error training model for {n_categories} categories with {feature_type}: {str(e)}")
            raise
    
    def train_all_categories(self, feature_types=None):
        """Train models on all category sizes with specified feature types"""
        # Use config feature types if none specified
        if feature_types is None:
            feature_types = DL_CONFIG['feature_types']
            
        logger.info("Starting DL model training for all categories...")
        
        all_results = {}
        
        print(f"\n{'='*80}")
        print(f"STARTING DL MODEL TRAINING PIPELINE")
        print(f"{'='*80}")
        print(f"Category sizes: {CATEGORY_SIZES}")
        print(f"Feature types: {feature_types}")
        print(f"Models: {DL_CONFIG['models']}")
        print(f"{'='*80}")
        
        for feature_type in feature_types:
            all_results[feature_type] = {}
            print(f"\n{'-'*60}")
            print(f"TRAINING WITH {feature_type.upper()} FEATURES")
            print(f"{'-'*60}")
            
            for n_categories in CATEGORY_SIZES:
                print(f"\n>>> Processing top_{n_categories}_categories with {feature_type.upper()} features...")
                
                try:
                    results = self.train_model_on_category(n_categories, feature_type)
                    all_results[feature_type][n_categories] = results
                    
                    # Save individual results using config paths
                    category_dir = SAVED_MODELS_CONFIG['dl_models_path'] / f'top_{n_categories}_categories'
                    category_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save as JSON
                    results_json = category_dir / f'dl_results_{feature_type}.json'
                    with open(results_json, 'w') as f:
                        json_safe_results = self.make_json_serializable(results)
                        json.dump(json_safe_results, f, indent=2)
                    
                    # Save as YAML
                    results_yaml = category_dir / f'dl_results_{feature_type}.yaml'
                    with open(results_yaml, 'w') as f:
                        yaml_safe_results = self.make_json_serializable(results)
                        # Remove large arrays for YAML (keep only summary metrics)
                        yaml_results = {k: v for k, v in yaml_safe_results.items() 
                                      if k not in ['predictions', 'probabilities', 'confusion_matrix']}
                        yaml.dump(yaml_results, f, default_flow_style=False)
                    
                    logger.info(f"✅ Results saved to {results_json} and {results_yaml}")
                    logger.info(f"✅ Training completed successfully for {n_categories} categories")
                
                except Exception as e:
                    logger.error(f"❌ Error training for {n_categories} categories: {str(e)}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    continue
                
                # Clear memory after each training
                tf.keras.backend.clear_session()
        
        print(f"\n{'='*80}")
        print(f"DL MODEL TRAINING PIPELINE COMPLETED")
        print(f"{'='*80}")
        
        # Print summary of saved data
        print(f"\nDL Final Results Summary:")
        print(f"  File location: {SAVED_MODELS_CONFIG['dl_models_path']}/dl_final_results.pkl")
        print(f"  Categories processed: {list(self.dl_final_results.keys())}")
        print(f"  Total model entries: {sum(len(models) for models in self.dl_final_results.values())}")
        
        return all_results
    
    def create_comprehensive_comparison(self, all_results):
        """Create comprehensive comparison plots and analysis"""
        try:
            logger.info("Creating comprehensive performance analysis...")
            
            # Use config path instead of hard-coded path
            comparison_dir = RESULTS_CONFIG['dl_comparisons_path']
            comparison_dir.mkdir(parents=True, exist_ok=True)
            
            # Performance comparison across categories and features
            performance_data = []
            
            for feature_type, feature_results in all_results.items():
                for n_categories, results in feature_results.items():
                    performance_data.append({
                        'feature_type': feature_type,
                        'n_categories': n_categories,
                        'top1_accuracy': results['top1_accuracy'],
                        'top3_accuracy': results['top3_accuracy'],
                        'top5_accuracy': results['top5_accuracy'],
                        'macro_f1': results['macro_f1'],
                        'micro_f1': results['micro_f1'],
                        'training_time': results['training_time'],
                        'inference_time': results['inference_time']
                    })
            
            df_performance = pd.DataFrame(performance_data)
            
            # Save performance comparison CSV
            performance_csv = comparison_dir / 'dl_performance_comparison.csv'
            df_performance.to_csv(performance_csv, index=False)
            
            # Create comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            
            # Plot 1: Top-1 Accuracy comparison
            for feature_type in df_performance['feature_type'].unique():
                data = df_performance[df_performance['feature_type'] == feature_type]
                axes[0, 0].plot(data['n_categories'], data['top1_accuracy'], 
                               marker='o', linewidth=2, label=f'{feature_type.upper()}')
            
            axes[0, 0].set_title('Top-1 Accuracy Comparison Across Categories')
            axes[0, 0].set_xlabel('Number of Categories')
            axes[0, 0].set_ylabel('Top-1 Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Macro F1 comparison
            for feature_type in df_performance['feature_type'].unique():
                data = df_performance[df_performance['feature_type'] == feature_type]
                axes[0, 1].plot(data['n_categories'], data['macro_f1'], 
                               marker='s', linewidth=2, label=f'{feature_type.upper()}')
            
            axes[0, 1].set_title('Macro F1 Score Comparison Across Categories')
            axes[0, 1].set_xlabel('Number of Categories')
            axes[0, 1].set_ylabel('Macro F1 Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Training time comparison
            for feature_type in df_performance['feature_type'].unique():
                data = df_performance[df_performance['feature_type'] == feature_type]
                axes[1, 0].bar(data['n_categories'] + (0.1 if feature_type == 'tfidf' else -0.1), 
                              data['training_time'], width=0.2, label=f'{feature_type.upper()}', alpha=0.8)
            
            axes[1, 0].set_title('Training Time Comparison Across Categories')
            axes[1, 0].set_xlabel('Number of Categories')
            axes[1, 0].set_ylabel('Training Time (seconds)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Top-K accuracy comparison (for largest category)
            largest_category = max(df_performance['n_categories'])
            largest_data = df_performance[df_performance['n_categories'] == largest_category]
            
            x_pos = range(len(largest_data))
            width = 0.25
            
            axes[1, 1].bar([p - width for p in x_pos], largest_data['top1_accuracy'], 
                          width, label='Top-1', alpha=0.8)
            axes[1, 1].bar(x_pos, largest_data['top3_accuracy'], 
                          width, label='Top-3', alpha=0.8)
            axes[1, 1].bar([p + width for p in x_pos], largest_data['top5_accuracy'], 
                          width, label='Top-5', alpha=0.8)
            
            axes[1, 1].set_title(f'Top-K Accuracy Comparison ({largest_category} Categories)')
            axes[1, 1].set_xlabel('Feature Type')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels([f.upper() for f in largest_data['feature_type']])
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save comparison plot
            comparison_plot = comparison_dir / 'dl_performance_comparison.png'
            plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance comparison plot saved: {comparison_plot}")
            
            return {
                'performance_csv': str(performance_csv),
                'comparison_plot': str(comparison_plot),
                'performance_data': performance_data
            }
            
        except Exception as e:
            logger.error(f"Error creating comprehensive comparison: {str(e)}")
            return {}
    
    def create_model_summary(self, all_results):
        """Create a comprehensive summary of model performance"""
        try:
            summary = {
                'training_summary': {},
                'performance_comparison': {},
                'training_curves': {},
                'best_performing_models': {},
                'category_analysis': {}
            }
            
            # Process results for each feature type
            for feature_type, feature_results in all_results.items():
                summary['training_summary'][feature_type] = {}
                summary['training_curves'][feature_type] = {}
                
                for n_categories, model_results in feature_results.items():
                    summary['training_summary'][feature_type][n_categories] = {
                        'top1_accuracy': model_results['top1_accuracy'],
                        'top3_accuracy': model_results['top3_accuracy'],
                        'top5_accuracy': model_results['top5_accuracy'],
                        'macro_f1': model_results['macro_f1'],
                        'micro_f1': model_results['micro_f1'],
                        'training_time': model_results['training_time'],
                        'inference_time': model_results['inference_time']
                    }
                    
                    summary['training_curves'][feature_type][n_categories] = model_results['training_history']
            
            # Find best performing models
            best_models = {}
            for feature_type, feature_results in all_results.items():
                best_accuracy = 0
                best_category = None
                best_result = None
                
                for n_categories, results in feature_results.items():
                    if results['top1_accuracy'] > best_accuracy:
                        best_accuracy = results['top1_accuracy']
                        best_category = n_categories
                        best_result = results
                
                if best_result:
                    best_models[feature_type] = {
                        'n_categories': best_category,
                        'top1_accuracy': best_accuracy,
                        'macro_f1': best_result['macro_f1'],
                        'training_time': best_result['training_time']
                    }
            
            summary['best_performing_models'] = best_models
            
            # Category-wise analysis
            for n_categories in CATEGORY_SIZES:
                if any(n_categories in results for results in all_results.values()):
                    category_comparison = {}
                    
                    for feature_type, feature_results in all_results.items():
                        if n_categories in feature_results:
                            result = feature_results[n_categories]
                            category_comparison[feature_type] = {
                                'top1_accuracy': result['top1_accuracy'],
                                'macro_f1': result['macro_f1'],
                                'training_time': result['training_time']
                            }
                    
                    summary['category_analysis'][n_categories] = category_comparison
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating model summary: {str(e)}")
            return {}
    
    
    def plot_and_analyze_dl_results(self, results_file_path=None, charts_dir=None):
        """
        Generate comprehensive plots and analysis for DL models only
        
        Args:
            results_file_path (str, optional): Path to the dl_final_results.pkl file. 
                                             If None, uses config path.
            charts_dir (str, optional): Directory to save charts. 
                                       If None, uses config path.
        """
        
        # Use config paths instead of hard-coded ones
        if results_file_path is None:
            results_file_path = RESULTS_CONFIG["dl_comparisons_path"] / "dl_final_results.pkl"
        else:
            results_file_path = Path(results_file_path)
        
        if charts_dir is None:
            charts_dir = RESULTS_CONFIG["dl_comparisons_path"] / "charts"
        else:
            charts_dir = Path(charts_dir)
        
        if not results_file_path.exists():
            print(f"No DL results file found at: {results_file_path}")
            return
        
        # Create charts directory
        charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        with open(results_file_path, "rb") as f:
            dl_final_results = pickle.load(f)
        
        print(f"Generating plots and analysis for DL results...")
        print(f"Results loaded from: {results_file_path}")
        print(f"Charts will be saved to: {charts_dir}")
        
        model_metrics = {}

        # Parse results and organize by model and feature type
        for n, results in dl_final_results.items():
            for entry in results:
                model = entry['model']
                feature_type = entry['feature_type']   # "tfidf" or "sbert"
                
                if model not in model_metrics:
                    model_metrics[model] = {}
                
                if feature_type not in model_metrics[model]:
                    model_metrics[model][feature_type] = {
                        'n': [], 
                        'accuracy': [], 
                        'precision': [], 
                        'recall': [], 
                        'f1_score': [],
                        # DL-specific metrics
                        'top1_accuracy': [],
                        'top3_accuracy': [],
                        'top5_accuracy': [],
                        'training_time': [],
                        'inference_time': []
                    }
                
                model_metrics[model][feature_type]['n'].append(entry['n_categories'])
                model_metrics[model][feature_type]['accuracy'].append(entry['accuracy'])
                model_metrics[model][feature_type]['precision'].append(entry['precision'])
                model_metrics[model][feature_type]['recall'].append(entry['recall'])
                model_metrics[model][feature_type]['f1_score'].append(entry['f1_score'])
                
                # Add DL-specific metrics
                model_metrics[model][feature_type]['top1_accuracy'].append(entry.get('top1_accuracy', entry['accuracy']))
                model_metrics[model][feature_type]['top3_accuracy'].append(entry.get('top3_accuracy', 0))
                model_metrics[model][feature_type]['top5_accuracy'].append(entry.get('top5_accuracy', 0))
                model_metrics[model][feature_type]['training_time'].append(entry.get('training_time', 0))
                model_metrics[model][feature_type]['inference_time'].append(entry.get('inference_time', 0))
            
        # =================================================================
        # LINE PLOTS - Performance vs Category Size
        # =================================================================
        def plot_metric(metric_name, ylabel=None):
            plt.figure(figsize=(12, 6))
            
            for model, features in model_metrics.items():
                for feature_type, data in features.items():
                    label = f"{model} ({feature_type.upper()})"
                    plt.plot(data['n'], data[metric_name], marker='o', label=label, linewidth=2)
            
            plt.title(f'{ylabel or metric_name.replace("_", " ").title()} vs Number of Web Service Categories (DL Models)')
            plt.xlabel('Number of Web Service Categories')
            plt.ylabel(ylabel or metric_name.replace("_", " ").title())
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plot_path = charts_dir / f"DL_Model_Performance_{metric_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"DL line plot saved: {plot_path}")
            plt.close()

        # Generate line plots for all metrics including top-K
        print("\nGenerating DL line plots...")
        metrics_config = {
            'accuracy': 'Accuracy',
            'precision': 'Precision (Macro)',
            'recall': 'Recall (Macro)',
            'f1_score': 'F1-Score (Macro)',
            'top1_accuracy': 'Top-1 Accuracy',
            'top3_accuracy': 'Top-3 Accuracy', 
            'top5_accuracy': 'Top-5 Accuracy',
            'training_time': 'Training Time (seconds)',
            'inference_time': 'Inference Time (seconds)'
        }
        
        for metric, ylabel in metrics_config.items():
            plot_metric(metric, ylabel)
        
        # =================================================================
        # COMBINED TOP-K ACCURACY PLOT
        # =================================================================
        print("\nGenerating combined DL top-K accuracy plot...")
        plt.figure(figsize=(14, 8))
        
        for model, features in model_metrics.items():
            for feature_type, data in features.items():
                label_base = f"{model} ({feature_type.upper()})"
                
                plt.plot(data['n'], data['top1_accuracy'], marker='o', label=f"{label_base} - Top-1", linewidth=2)
                plt.plot(data['n'], data['top3_accuracy'], marker='s', label=f"{label_base} - Top-3", linewidth=2, linestyle='--')
                plt.plot(data['n'], data['top5_accuracy'], marker='^', label=f"{label_base} - Top-5", linewidth=2, linestyle=':')
        
        plt.title('DL Models: Top-K Accuracy Comparison')
        plt.xlabel('Number of Web Service Categories')
        plt.ylabel('Top-K Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plot_path = charts_dir / "DL_Model_Performance_topk_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Combined DL top-K plot saved: {plot_path}")
        plt.close()
        
        # =================================================================
        # TRAINING TIME ANALYSIS
        # =================================================================
        print("\nGenerating DL training time analysis...")
        plt.figure(figsize=(12, 8))
        
        for model, features in model_metrics.items():
            for feature_type, data in features.items():
                label = f"{model} ({feature_type.upper()})"
                plt.semilogy(data['n'], data['training_time'], marker='o', label=label, linewidth=2)
        
        plt.title('DL Models: Training Time vs Number of Categories')
        plt.xlabel('Number of Web Service Categories')
        plt.ylabel('Training Time (seconds, log scale)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plot_path = charts_dir / "DL_Model_Performance_training_time.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"DL training time plot saved: {plot_path}")
        plt.close()
        
        # =================================================================
        # BAR PLOTS - Performance by Category Size (Enhanced)
        # =================================================================
        print("\nGenerating enhanced DL bar plots for each category size...")
        
        for n in CATEGORY_SIZES:
            if n not in dl_final_results:
                print(f"Skipping n={n} (no DL results found)")
                continue

            combined_results = dl_final_results[n]
            df_combined = pd.DataFrame(combined_results)

            # Enhanced metrics selection including top-K
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'top1_accuracy', 'top3_accuracy', 'top5_accuracy']
            available_metrics = [col for col in metrics_to_plot if col in df_combined.columns]
            
            df_plot_combined = df_combined[['model'] + available_metrics]
            df_plot_combined.set_index('model', inplace=True)

            # Main performance plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Standard metrics
            standard_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            df_plot_combined[standard_metrics].plot(kind='bar', ax=ax1, width=0.8)
            ax1.set_title(f'DL Standard Performance Metrics - Top {n} Categories', fontsize=14)
            ax1.set_ylabel('Score')
            ax1.set_ylim(0.3, 1.0)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3)
            ax1.legend(title='Metric')
            
            # Top-K accuracy metrics
            topk_metrics = ['top1_accuracy', 'top3_accuracy', 'top5_accuracy']
            available_topk = [col for col in topk_metrics if col in df_plot_combined.columns]
            if available_topk:
                df_plot_combined[available_topk].plot(kind='bar', ax=ax2, width=0.8)
                ax2.set_title(f'DL Top-K Accuracy Metrics - Top {n} Categories', fontsize=14)
                ax2.set_ylabel('Top-K Accuracy')
                ax2.set_ylim(0.3, 1.0)
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(axis='y', alpha=0.3)
                ax2.legend(title='Top-K Metric')
            
            plt.tight_layout()
            
            # Save enhanced plot
            plot_path = charts_dir / f"DL_Model_Performance_enhanced_top_{n}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Enhanced DL bar plot saved: {plot_path}")
            plt.close()

        # =================================================================
        # FEATURE TYPE COMPARISON
        # =================================================================
        print("\nGenerating DL feature type comparison...")
        
        # Aggregate performance by feature type across all categories
        feature_performance = {}
        for n, results in dl_final_results.items():
            for entry in results:
                feature_type = entry['feature_type']
                if feature_type not in feature_performance:
                    feature_performance[feature_type] = {
                        'accuracy': [],
                        'f1_score': [],
                        'top1_accuracy': [],
                        'training_time': []
                    }
                
                feature_performance[feature_type]['accuracy'].append(entry['accuracy'])
                feature_performance[feature_type]['f1_score'].append(entry['f1_score'])
                feature_performance[feature_type]['top1_accuracy'].append(entry.get('top1_accuracy', entry['accuracy']))
                feature_performance[feature_type]['training_time'].append(entry.get('training_time', 0))
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        feature_types = list(feature_performance.keys())
        metrics = ['accuracy', 'f1_score', 'top1_accuracy', 'training_time']
        titles = ['Average Accuracy', 'Average F1-Score', 'Average Top-1 Accuracy', 'Average Training Time']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            means = [np.mean(feature_performance[ft][metric]) for ft in feature_types]
            stds = [np.std(feature_performance[ft][metric]) for ft in feature_types]
            
            bars = ax.bar(feature_types, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_title(f'DL Models: {title} by Feature Type')
            ax.set_ylabel(title.split()[-1])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_path = charts_dir / "DL_Feature_Type_Comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"DL feature comparison plot saved: {plot_path}")
        plt.close()

        # =================================================================
        # SUMMARY STATISTICS TABLE
        # =================================================================
        print("\nGenerating DL summary statistics...")
        
        # Create summary table
        summary_data = []
        for n in CATEGORY_SIZES:
            if n in dl_final_results:
                for entry in dl_final_results[n]:
                    summary_data.append({
                        'Categories': n,
                        'Model': entry['model'],
                        'Feature': entry['feature_type'],
                        'Accuracy': entry['accuracy'],
                        'F1-Score': entry['f1_score'],
                        'Top-1': entry.get('top1_accuracy', entry['accuracy']),
                        'Top-3': entry.get('top3_accuracy', 0),
                        'Top-5': entry.get('top5_accuracy', 0),
                        'Training Time': entry.get('training_time', 0),
                        'Inference Time': entry.get('inference_time', 0)
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.round(4)
        
        # Save summary table
        summary_path = charts_dir / "DL_Model_Performance_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"DL summary table saved: {summary_path}")
        
        # Display best performing models
        print("\nTop performing DL models by metric:")
        for metric in ['Accuracy', 'F1-Score', 'Top-1', 'Top-3', 'Top-5']:
            if metric in summary_df.columns:
                best = summary_df.loc[summary_df[metric].idxmax()]
                print(f"  {metric}: {best['Model']} ({best['Feature']}) on {best['Categories']} categories = {best[metric]:.4f}")
        
        # Best model overall
        best_overall = summary_df.loc[summary_df['Top-1'].idxmax()]
        print(f"\nBest Overall DL Model:")
        print(f"  {best_overall['Model']} ({best_overall['Feature']}) on {best_overall['Categories']} categories")
        print(f"  Top-1 Accuracy: {best_overall['Top-1']:.4f}")
        print(f"  F1-Score: {best_overall['F1-Score']:.4f}")
        print(f"  Training Time: {best_overall['Training Time']:.2f}s")

    def plot_dl_results_only(self):
        """Convenience function to plot DL results with config paths"""
        self.plot_and_analyze_dl_results()  # Now uses config paths by default

def main():
    """Main function to run comprehensive DL model training and analysis"""
    pass

if __name__ == "__main__":
    main()