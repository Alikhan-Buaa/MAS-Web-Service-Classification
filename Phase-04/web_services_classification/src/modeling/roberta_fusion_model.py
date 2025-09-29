"""
Enhanced RoBERTa Fusion Models for Web Service Classification
Fusion model using multiple RoBERTa layers with standardized naming
"""

import pandas as pd
import numpy as np
import logging
import json
import time
import traceback
import random
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import RobertaTokenizer, RobertaModel

# Import configuration and utilities
from src.config import (
    CATEGORY_SIZES, SAVED_MODELS_CONFIG, FUSION_CONFIG, 
    PREPROCESSING_CONFIG, RANDOM_SEED, RESULTS_CONFIG
)
from src.evaluation.evaluate import ModelEvaluator
from src.utils.utils import FileNamingStandard

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================================
# ROBERTA FUSION MODEL ARCHITECTURE
# ============================================================================

class RoBERTaFusionModel(nn.Module):
    """
    RoBERTa Fusion Model - extracts and fuses embeddings from multiple layers
    """
    
    def __init__(self, config, num_labels):
        super(RoBERTaFusionModel, self).__init__()
        
        self.config = config
        self.num_labels = num_labels
        self.fusion_type = config.get('fusion_type', 'concat')
        self.num_layers_to_fuse = config.get('num_layers_to_fuse', 4)
        dropout = config.get('dropout', 0.3)
        
        # Load RoBERTa
        roberta_model_name = config.get('roberta_model', 'roberta-base')
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.hidden_size = self.roberta.config.hidden_size
        
        # Calculate fused dimension
        if self.fusion_type == 'concat':
            fused_dim = self.hidden_size * self.num_layers_to_fuse
        elif self.fusion_type == 'average':
            fused_dim = self.hidden_size
        elif self.fusion_type == 'weighted':
            self.layer_weights = nn.Parameter(torch.ones(self.num_layers_to_fuse))
            fused_dim = self.hidden_size
        elif self.fusion_type == 'gating':
            self.gate = nn.Sequential(
                nn.Linear(self.hidden_size * self.num_layers_to_fuse, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, self.num_layers_to_fuse),
                nn.Softmax(dim=-1)
            )
            fused_dim = self.hidden_size
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )
        
        # Temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1))
    
    def extract_layer_embeddings(self, input_ids, attention_mask):
        """Extract [CLS] embeddings from multiple layers"""
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        all_hidden_states = outputs.hidden_states
        layer_embeddings = []
        
        for i in range(-self.num_layers_to_fuse, 0):
            cls_embedding = all_hidden_states[i][:, 0, :]
            layer_embeddings.append(cls_embedding)
        
        return layer_embeddings
    
    def fuse_embeddings(self, layer_embeddings):
        """Fuse layer embeddings"""
        if self.fusion_type == 'concat':
            return torch.cat(layer_embeddings, dim=1)
        
        elif self.fusion_type == 'average':
            stacked = torch.stack(layer_embeddings, dim=0)
            return torch.mean(stacked, dim=0)
        
        elif self.fusion_type == 'weighted':
            weights = F.softmax(self.layer_weights, dim=0)
            weighted_layers = [weights[i] * emb for i, emb in enumerate(layer_embeddings)]
            return torch.stack(weighted_layers, dim=0).sum(dim=0)
        
        elif self.fusion_type == 'gating':
            concat = torch.cat(layer_embeddings, dim=1)
            gate_weights = self.gate(concat)
            weighted_layers = [gate_weights[:, i:i+1] * emb for i, emb in enumerate(layer_embeddings)]
            return torch.stack(weighted_layers, dim=0).sum(dim=0)
    
    def forward(self, input_ids, attention_mask, apply_temperature=False):
        """Forward pass"""
        layer_embeddings = self.extract_layer_embeddings(input_ids, attention_mask)
        fused_embedding = self.fuse_embeddings(layer_embeddings)
        logits = self.classifier(fused_embedding)
        
        if apply_temperature:
            logits = logits / self.temperature
        
        return logits


# ============================================================================
# DATASET
# ============================================================================

class RoBERTaFusionDataset(Dataset):
    """Dataset for RoBERTa fusion model"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# TRAINER
# ============================================================================

class RoBERTaFusionTrainer:
    """Enhanced RoBERTa Fusion model trainer with standardized naming"""
    
    @staticmethod
    def make_json_serializable(obj):
        """Convert numpy types and Path objects to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: RoBERTaFusionTrainer.make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [RoBERTaFusionTrainer.make_json_serializable(item) for item in obj]
        elif isinstance(obj, (Path, type(Path()))):
            return str(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        
        # Copy config and add missing values with defaults if not present
        self.config = FUSION_CONFIG.copy()
        if 'weight_decay' not in self.config:
            self.config['weight_decay'] = 0.01
        if 'eval_batch_size' not in self.config:
            self.config['eval_batch_size'] = 32
        if 'gradient_clip' not in self.config:
            self.config['gradient_clip'] = 1.0
        if 'scheduler' not in self.config:
            self.config['scheduler'] = {
                'mode': 'max',
                'patience': 2,
                'factor': 0.5,
                'verbose': True
            }
        
        self.evaluator = ModelEvaluator()
        
        # Configure GPU
        self._configure_gpu()
        
        # Create results directories
        self._create_directories()
    
    def _configure_gpu(self):
        """Configure GPU memory and device"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                logger.info("Using CPU")
        except Exception as e:
            logger.warning(f"GPU configuration warning: {e}")
            self.device = torch.device("cpu")
    
    def _create_directories(self):
        """Create necessary directories for results and visualizations"""
        directories = [
            RESULTS_CONFIG['fusion_results_path'],
            RESULTS_CONFIG['fusion_comparisons_path'],
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create category-specific directories
        for n_categories in CATEGORY_SIZES:
            category_dir = RESULTS_CONFIG['fusion_category_paths'][n_categories]
            category_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Created result directories for RoBERTa Fusion models")
    
    def get_model_config(self, fusion_type):
        """Get fusion-specific configuration"""
        model_config = self.config.copy()
        model_config['fusion_type'] = fusion_type
        return model_config
    
    def load_tokenizer(self, model_name=None):
        """Load RoBERTa tokenizer"""
        try:
            if model_name is None:
                model_name = self.config.get('roberta_model', 'roberta-base')
            
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            logger.info(f"Loaded RoBERTa tokenizer: {model_name}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def prepare_datasets(self, n_categories):
        """Load and prepare datasets for training"""
        try:
            logger.info(f"Loading datasets for top_{n_categories}_categories")
            
            # Load datasets using correct config paths
            splits_dir = Path(PREPROCESSING_CONFIG["splits"].format(n=n_categories))
            if not splits_dir.exists():
                raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
            
            train_df = pd.read_csv(splits_dir / 'train.csv')
            val_df = pd.read_csv(splits_dir / 'val.csv')
            test_df = pd.read_csv(splits_dir / 'test.csv')
            
            logger.info(f"Loaded datasets - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            
            # Use cleaned_text if available, otherwise use original text
            text_column = 'cleaned_text' if 'cleaned_text' in train_df.columns else 'text'
            if text_column not in train_df.columns:
                text_column = 'Service Description' if 'Service Description' in train_df.columns else train_df.columns[0]
            
            logger.info(f"Using text column: {text_column}")
            
            max_length = self.config.get('max_length', 128)
            
            # Create datasets
            train_dataset = RoBERTaFusionDataset(
                train_df[text_column].astype(str).tolist(),
                train_df['encoded_label'].tolist(),
                self.tokenizer,
                max_length
            )
            
            val_dataset = RoBERTaFusionDataset(
                val_df[text_column].astype(str).tolist(),
                val_df['encoded_label'].tolist(),
                self.tokenizer,
                max_length
            )
            
            test_dataset = RoBERTaFusionDataset(
                test_df[text_column].astype(str).tolist(),
                test_df['encoded_label'].tolist(),
                self.tokenizer,
                max_length
            )
            
            logger.info("Datasets prepared successfully")
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Error preparing datasets: {e}")
            raise
    
    def create_model(self, num_labels, fusion_config):
        """Create RoBERTa Fusion model"""
        try:
            self.model = RoBERTaFusionModel(fusion_config, num_labels)
            fusion_type = fusion_config.get('fusion_type', 'concat')
            layers = fusion_config.get('num_layers_to_fuse', 4)
            logger.info(f"Created RoBERTa Fusion model: {fusion_type}, layers={layers}, labels={num_labels}")
            return self.model
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
    
    def train_epoch(self, model, dataloader, optimizer, criterion):
        """Train one epoch"""
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.get('gradient_clip', 1.0))
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def evaluate_epoch(self, model, dataloader, criterion):
        """Evaluate one epoch"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                probs = F.softmax(logits, dim=1)
                
                total_loss += loss.item()
                all_probs.append(probs.cpu())
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        all_probs = torch.cat(all_probs, dim=0).numpy()
        
        return avg_loss, accuracy, all_preds, all_labels, all_probs
    
    def calibrate_temperature(self, model, val_loader):
        """Temperature scaling calibration"""
        logger.info("Calibrating temperature...")
        model.eval()
        
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = model(input_ids, attention_mask, apply_temperature=False)
                logits_list.append(logits)
                labels_list.append(labels)
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([model.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(logits / model.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        logger.info(f"Optimal temperature: {model.temperature.item():.4f}")
    
    def plot_training_history(self, history, model_name, n_categories):
        """Create training history plots with standardized naming"""
        try:
            if not history['train_loss'] or not history['val_loss']:
                logger.warning("Insufficient training history for plotting")
                return None
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            epochs = range(1, len(history['train_loss']) + 1)
            
            # Plot loss
            ax1.plot(epochs, history['train_loss'], label='Training Loss', linewidth=2, marker='o')
            ax1.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2, marker='s')
            ax1.set_title(f'{model_name} - Training & Validation Loss\n{n_categories} Categories')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot accuracy
            if history['val_acc']:
                ax2.plot(epochs, history['train_acc'], label='Training Accuracy', linewidth=2, marker='o')
                ax2.plot(epochs, history['val_acc'], label='Validation Accuracy', linewidth=2, marker='s')
                ax2.set_title(f'{model_name} - Training & Validation Accuracy\n{n_categories} Categories')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot using standardized naming
            plot_dir = RESULTS_CONFIG['fusion_category_paths'][n_categories]
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            filename = FileNamingStandard.generate_training_history_filename(model_name, n_categories)
            plot_file = plot_dir / filename
            
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training history plot saved: {plot_file}")
            return str(plot_file)
            
        except Exception as e:
            logger.error(f"Error creating training history plot: {str(e)}")
            return None
    
    def evaluate_fusion_model(self, model, test_loader, model_name, n_categories, class_labels):
        """Comprehensive evaluation of RoBERTa Fusion model"""
        try:
            logger.info(f"Evaluating model: {model_name}")
            
            model.eval()
            all_preds = []
            all_labels = []
            all_probs = []
            
            start_time = time.time()
            
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    logits = model(input_ids, attention_mask, apply_temperature=True)
                    probs = F.softmax(logits, dim=1)
                    
                    all_probs.append(probs.cpu())
                    preds = torch.argmax(probs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            inference_time = time.time() - start_time
            
            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            y_proba = torch.cat(all_probs, dim=0).numpy()
            
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
            
            # Top-K accuracies using common evaluator
            y_true_onehot = np.eye(n_categories)[y_true]
            top1_accuracy = self.evaluator.calculate_top_k_accuracy(y_true_onehot, y_proba, k=1)
            top3_accuracy = self.evaluator.calculate_top_k_accuracy(y_true_onehot, y_proba, k=3)
            top5_accuracy = self.evaluator.calculate_top_k_accuracy(y_true_onehot, y_proba, k=5)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create visualizations using common evaluator with standardized naming
            fusion_type = model_name.split('-')[-1].lower()
            cm_plot_path = self.evaluator.generate_confusion_heatmap(
                cm, class_labels, model_name, n_categories, f"fusion_{fusion_type}", "fusion"
            )
            report_path = self.evaluator.generate_classification_report_csv(
                y_true, y_pred, class_labels, model_name, n_categories, f"fusion_{fusion_type}", "fusion"
            )
            
            # Compile results
            results = {
                'model_name': model_name,
                'feature_type': f'fusion_{fusion_type}',
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
                'confusion_matrix_plot': cm_plot_path,
                'classification_report_path': str(report_path),
                'inference_time': float(inference_time)
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
    
    def train_model_on_category(self, n_categories, fusion_type=None):
        """Train RoBERTa Fusion model on a specific category size"""
        try:
            if fusion_type is None:
                fusion_type = 'concat'
            
            # Validate fusion type
            available_types = self.config.get('fusion_types', ['concat', 'average', 'weighted', 'gating'])
            if fusion_type not in available_types:
                raise ValueError(f"Fusion type {fusion_type} not in available types: {available_types}")
            
            model_name = f"RoBERTa-Fusion-{fusion_type.capitalize()}"
            logger.info(f"Training {model_name} for top_{n_categories}_categories")
            
            # Clear any existing model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Get model-specific configuration
            model_config = self.get_model_config(fusion_type)
            
            # Load tokenizer
            roberta_model = self.config.get('roberta_model', 'roberta-base')
            self.load_tokenizer(roberta_model)
            
            # Prepare datasets
            train_dataset, val_dataset, test_dataset = self.prepare_datasets(n_categories)
            
            # Load class labels using common evaluator
            class_labels = self.evaluator.load_class_labels(n_categories)
            
            # Create model
            model = self.create_model(n_categories, model_config).to(self.device)
            
            # Training setup
            batch_size = self.config.get('batch_size', 16)
            eval_batch_size = self.config.get('eval_batch_size', 32)
            learning_rate = self.config.get('learning_rate', 2e-5)
            weight_decay = self.config.get('weight_decay', 0.01)
            epochs = self.config.get('num_train_epochs', 10)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()
            
            scheduler_config = self.config.get('scheduler', {})
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.get('mode', 'max'),
                patience=scheduler_config.get('patience', 2),
                factor=scheduler_config.get('factor', 0.5),
                verbose=scheduler_config.get('verbose', True)
            )
            
            # Training history
            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
            best_val_acc = 0.0
            best_model_state = None
            
            # Train model
            print(f"\nTraining {model_name} on top_{n_categories}_categories...")
            print(f"Batch size: Train={batch_size}, Eval={eval_batch_size}")
            print(f"Learning rate: {learning_rate}")
            
            start_time = time.time()
            
            for epoch in range(epochs):
                train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc, _, _, _ = self.evaluate_epoch(model, val_loader, criterion)
                
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                scheduler.step(val_acc)
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                print(f"  LR: {current_lr:.2e}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                    print(f"  ✓ Best model (Val Acc: {val_acc:.4f})")
                print()
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Load best model
            model.load_state_dict(best_model_state)
            
            # Save model with standardized naming
            model_dir = SAVED_MODELS_CONFIG['fusion_models_path'] / f'top_{n_categories}_categories'
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_filename = FileNamingStandard.generate_model_filename(
                model_name, f'fusion_{fusion_type}', n_categories, 'model'
            )
            model_path = model_dir / model_filename
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'fusion_type': fusion_type,
                'num_layers_to_fuse': model_config.get('num_layers_to_fuse', 4),
                'n_categories': n_categories,
                'roberta_model': roberta_model,
                'config': model_config
            }, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Calibrate temperature
            self.calibrate_temperature(model, val_loader)
            
            # Create training history plot
            display_name = FileNamingStandard.standardize_model_name(model_name)
            history_plot_path = self.plot_training_history(history, display_name, n_categories)
            
            # Evaluate model
            eval_results = self.evaluate_fusion_model(model, test_loader, display_name, n_categories, class_labels)
            eval_results['training_time'] = float(training_time)
            eval_results['model_path'] = str(model_path)
            eval_results['model_variant'] = fusion_type
            eval_results['training_history_plot'] = history_plot_path
            eval_results['batch_size'] = batch_size
            eval_results['learning_rate'] = learning_rate
            
            # Print metrics using common evaluator
            self.evaluator.print_model_metrics(eval_results, display_name, n_categories, f"fusion_{fusion_type}", training_time, "Fusion")
            
            # Save performance data using common evaluator  
            self.evaluator.save_model_performance_data(eval_results, display_name, n_categories, f"fusion_{fusion_type}", "fusion")
            
            return eval_results
            
        except Exception as e:
            logger.error(f"Error training {fusion_type} fusion for {n_categories} categories: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def save_results_for_overall_analysis(self, all_results):
        """Save results in the format expected by OverallPerformanceAnalyzer"""
        try:
            comparisons_path = RESULTS_CONFIG['fusion_comparisons_path']
            comparisons_path.mkdir(parents=True, exist_ok=True)
            
            # Transform results to match ML/DL/BERT format (list-based structure)
            formatted_results = {}
            
            for fusion_type, fusion_results in all_results.items():
                for n_categories, result in fusion_results.items():
                    if n_categories not in formatted_results:
                        formatted_results[n_categories] = []  # Use list like other models
                    
                    # Create result entry with 'model' key for compatibility
                    result_entry = result.copy()
                    
                    # Add 'model' key (required by plotting function)
                    result_entry['model'] = result.get('model_name', f'RoBERTa-Fusion-{fusion_type.capitalize()}')
                    
                    # Ensure all required keys are present
                    if 'n_categories' not in result_entry:
                        result_entry['n_categories'] = n_categories
                    
                    # Add to list
                    formatted_results[n_categories].append(result_entry)
            
            # Save as pickle file
            pickle_file = comparisons_path / "fusion_final_results.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(formatted_results, f)
            
            logger.info(f"Fusion results saved for overall analysis: {pickle_file}")
            
            # Also save JSON for debugging
            json_file = comparisons_path / "fusion_final_results.json"
            with open(json_file, 'w') as f:
                json_safe_results = self.make_json_serializable(formatted_results)
                json.dump(json_safe_results, f, indent=2)
            
            logger.info(f"Fusion results JSON saved: {json_file}")
            
        except Exception as e:
            logger.error(f"Error saving Fusion results for overall analysis: {e}")
    
    def train_fusion_models(self, categories=None):
        """Train all RoBERTa Fusion models from config"""
        if categories is None:
            categories = CATEGORY_SIZES
        
        logger.info("Training RoBERTa Fusion models from config")
        
        all_results = {}
        
        print(f"\n{'='*80}")
        print(f"STARTING RoBERTa FUSION MODEL TRAINING PIPELINE")
        print(f"{'='*80}")
        print(f"Category sizes: {categories}")
        print(f"Fusion types: {self.config.get('fusion_types', ['concat', 'average', 'weighted', 'gating'])}")
        print(f"{'='*80}")
        
        # Train models from config
        for fusion_type in self.config.get('fusion_types', ['concat', 'average', 'weighted', 'gating']):
            print(f"\n{'-'*60}")
            print(f"TRAINING FUSION-{fusion_type.upper()}")
            print(f"{'-'*60}")
            
            fusion_results = {}
            
            for n_categories in categories:
                print(f"\n>>> Processing top_{n_categories}_categories with {fusion_type} fusion...")
                
                try:
                    results = self.train_model_on_category(n_categories, fusion_type)
                    fusion_results[n_categories] = results
                    
                    # Save individual results
                    category_dir = SAVED_MODELS_CONFIG['fusion_models_path'] / f'top_{n_categories}_categories'
                    category_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save as JSON with standardized naming
                    results_json = category_dir / f'fusion_{fusion_type}_results.json'
                    with open(results_json, 'w') as f:
                        json_safe_results = self.make_json_serializable(results)
                        json.dump(json_safe_results, f, indent=2)
                    
                    logger.info(f"Results saved to {results_json}")
                    logger.info(f"Training completed successfully for {fusion_type} on {n_categories} categories")
                    
                except Exception as e:
                    logger.error(f"Error training {fusion_type} for {n_categories} categories: {str(e)}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    continue
                
                # Clear GPU memory after each training
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            all_results[fusion_type] = fusion_results
        
        print(f"\n{'='*80}")
        print(f"RoBERTa FUSION MODEL TRAINING PIPELINE COMPLETED")
        print(f"{'='*80}")
        
        # Print comparison if multiple models trained
        if len(all_results) > 1:
            self._print_fusion_comparison(all_results)
        
        # Save results for overall analysis
        self.save_results_for_overall_analysis(all_results)
        
        return all_results
    
    def _print_fusion_comparison(self, all_results):
        """Print comparison between RoBERTa Fusion models"""
        print(f"\n{'='*80}")
        print(f"RoBERTa FUSION MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        for n_categories in CATEGORY_SIZES:
            results_for_category = {}
            
            # Collect results for this category
            for fusion_key, fusion_results in all_results.items():
                if n_categories in fusion_results:
                    results_for_category[fusion_key] = fusion_results[n_categories]
            
            if results_for_category:
                print(f"\nTop {n_categories} Categories Results:")
                print(f"{'Fusion Type':<15} {'Top-1 Acc':<10} {'Top-3 Acc':<10} {'Top-5 Acc':<10} {'Macro F1':<10} {'Training Time':<15}")
                print("-" * 85)
                
                # Sort by F1 score
                fusion_scores = []
                for fusion_key, result in results_for_category.items():
                    fusion_scores.append((
                        fusion_key,
                        result['top1_accuracy'],
                        result['top3_accuracy'],
                        result['top5_accuracy'],
                        result['macro_f1'],
                        result['training_time']
                    ))
                
                fusion_scores.sort(key=lambda x: x[4], reverse=True)  # Sort by F1
                
                for fusion_type, top1, top3, top5, f1, time_taken in fusion_scores:
                    print(f"{fusion_type:<15} {top1:<10.4f} {top3:<10.4f} {top5:<10.4f} {f1:<10.4f} {time_taken:<15.2f}")
                
                # Performance comparison
                if len(fusion_scores) >= 2:
                    best_f1 = fusion_scores[0][4]
                    worst_f1 = fusion_scores[-1][4]
                    improvement = best_f1 - worst_f1
                    
                    print(f"\nPerformance Analysis:")
                    print(f"  Best fusion: {fusion_scores[0][0]} (F1: {best_f1:.4f})")
                    print(f"  F1 improvement over worst: {improvement:+.4f}")
                    print(f"  Relative improvement: {(improvement/worst_f1)*100:+.2f}%")
        
        print(f"{'='*80}")
    
    def train_all_categories(self):
        """Train RoBERTa Fusion models on all category sizes (uses config fusion types)"""
        return self.train_fusion_models()
    
    def plot_fusion_results_only(self):
        """Convenience function to plot Fusion results with config paths"""
        results_file_path = RESULTS_CONFIG["fusion_comparisons_path"] / "fusion_final_results.pkl"
        charts_dir = RESULTS_CONFIG["fusion_comparisons_path"] / "charts"
        
        self.evaluator.plot_results_comparison(results_file_path, charts_dir, "fusion")


def main():
    """Main function to run comprehensive RoBERTa Fusion model training and analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RoBERTa Fusion Model Training for Web Service Classification")
    parser.add_argument("--fusion-type", type=str, default="all", 
                       choices=["concat", "average", "weighted", "gating", "all"],
                       help="Fusion type to train (default: all)")
    parser.add_argument("--categories", nargs="+", type=int, default=CATEGORY_SIZES,
                       help="Category sizes to train")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (overrides config)")
    parser.add_argument("--layers", type=int, default=None,
                       help="Number of layers to fuse (overrides config)")
    
    args = parser.parse_args()
    
    trainer = RoBERTaFusionTrainer()
    
    # Override config if command line args provided
    if args.epochs is not None:
        trainer.config['num_train_epochs'] = args.epochs
        logger.info(f"Overriding epochs: {args.epochs}")
    if args.batch_size is not None:
        trainer.config['batch_size'] = args.batch_size
        logger.info(f"Overriding batch size: {args.batch_size}")
    if args.lr is not None:
        trainer.config['learning_rate'] = args.lr
        logger.info(f"Overriding learning rate: {args.lr}")
    if args.layers is not None:
        trainer.config['num_layers_to_fuse'] = args.layers
        logger.info(f"Overriding layers to fuse: {args.layers}")
    
    if args.fusion_type == "all":
        # Train all RoBERTa Fusion models from config
        results = trainer.train_fusion_models(args.categories)
    else:
        # Train single fusion type
        logger.info(f"Training single fusion type: {args.fusion_type}")
        
        results = {}
        for n_categories in args.categories:
            results[n_categories] = trainer.train_model_on_category(n_categories, args.fusion_type)
    
    # Save final results
    out_file = SAVED_MODELS_CONFIG["fusion_models_path"] / "fusion_final_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json_safe_results = trainer.make_json_serializable(results)
        json.dump(json_safe_results, f, indent=2)
    logger.info(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()