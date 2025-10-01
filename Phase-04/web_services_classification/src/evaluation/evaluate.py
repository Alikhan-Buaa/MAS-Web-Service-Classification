"""
Enhanced ModelEvaluator with standardized naming and proper model type handling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import logging
from pathlib import Path
from sklearn.metrics import classification_report

# Import standardized naming
from src.utils.utils import FileNamingStandard
from src.config import RESULTS_CONFIG, CATEGORY_SIZES

# Setup logging
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Enhanced evaluator with standardized naming across all model types"""
    
    def __init__(self):
        self.final_results = {}
        
    def calculate_top_k_accuracy(self, y_true, y_proba, k=1):
        """Calculate top-k accuracy"""
        try:
            # Handle different input formats
            if hasattr(y_true, 'shape') and len(y_true.shape) > 1:
                # One-hot encoded
                if y_true.shape[1] > 1:
                    y_true_labels = np.argmax(y_true, axis=1)
                else:
                    y_true_labels = y_true.flatten()
            else:
                # Already label indices
                y_true_labels = y_true
            
            # Get top-k predictions
            if k == 1:
                top_k_preds = np.argmax(y_proba, axis=1)
                return np.mean(top_k_preds == y_true_labels)
            else:
                top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
                return np.mean([label in pred for label, pred in zip(y_true_labels, top_k_preds)])
                
        except Exception as e:
            logger.error(f"Error calculating top-{k} accuracy: {e}")
            return 0.0
    
    def load_class_labels(self, n_categories):
        """Load class labels for a given category size"""
        try:
            from src.config import PREPROCESSING_CONFIG
            splits_dir = Path(PREPROCESSING_CONFIG["splits"].format(n=n_categories))
            
            # Try to load from train.csv
            train_df = pd.read_csv(splits_dir / 'train.csv')
            
            if 'label' in train_df.columns and 'encoded_label' in train_df.columns:
                # Create mapping from encoded labels to original labels
                label_mapping = train_df.groupby('encoded_label')['label'].first().sort_index()
                return label_mapping.tolist()
            else:
                # Fallback: create generic labels
                return [f"Category_{i}" for i in range(n_categories)]
                
        except Exception as e:
            logger.error(f"Error loading class labels for {n_categories} categories: {e}")
            return [f"Category_{i}" for i in range(n_categories)]
    
    def _get_results_path(self, model_type, n_categories):
        """Get the correct results path based on model type"""
        model_type_mapping = {
            'ml': 'ml',
            'dl': 'dl', 
            'bert': 'bert',
            'roberta': 'bert',  # Map roberta to bert directory
            'deepseek': 'deepseek',
            'fusion':'fusion'
        }
        
        # Get the correct model type for directory structure
        dir_model_type = model_type_mapping.get(model_type, model_type)
        
        # Determine the correct results path
        if dir_model_type == 'bert':
            return RESULTS_CONFIG['bert_category_paths'][n_categories]
        elif dir_model_type == 'deepseek':
            return RESULTS_CONFIG['deepseek_category_paths'][n_categories]
        elif dir_model_type == 'ml':
            return RESULTS_CONFIG['ml_category_paths'][n_categories]
        elif dir_model_type == 'dl':
            return RESULTS_CONFIG['dl_category_paths'][n_categories]
        elif dir_model_type == 'fusion':
            return RESULTS_CONFIG['fusion_category_paths'][n_categories]
            
        else:
            # Fallback - create a generic path
            fallback_path = Path(f"results/{dir_model_type}/top_{n_categories}_categories")
            fallback_path.mkdir(parents=True, exist_ok=True)
            return fallback_path
    
    def generate_confusion_heatmap(self, cm, class_labels, model_name, n_categories, feature_type, model_type):
        """Generate confusion matrix heatmap with standardized naming"""
        try:
            # Get results path
            results_path = self._get_results_path(model_type, n_categories)
            results_path.mkdir(parents=True, exist_ok=True)
            
            # Create confusion matrix plot
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels
            )
            plt.title(f'{model_name} - Confusion Matrix\n{n_categories} Categories ({feature_type.upper()})')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save plot using standardized filename
            filename = FileNamingStandard.generate_confusion_matrix_filename(
                model_name, feature_type, n_categories
            )
            plot_file = results_path / filename
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix saved: {plot_file}")
            return str(plot_file)
            
        except Exception as e:
            logger.error(f"Error generating confusion matrix for {model_name}: {e}")
            return None
    
    def generate_classification_report_csv(self, y_true, y_pred, class_labels, model_name, n_categories, feature_type, model_type):
        """Generate classification report CSV with standardized naming"""
        try:
            # Get results path
            results_path = self._get_results_path(model_type, n_categories)
            results_path.mkdir(parents=True, exist_ok=True)
            
            # Generate classification report
            report = classification_report(
                y_true, 
                y_pred, 
                target_names=class_labels,
                output_dict=True,
                zero_division=0
            )
            
            # Convert to DataFrame
            report_df = pd.DataFrame(report).transpose()
            
            # Save using standardized filename
            filename = FileNamingStandard.generate_classification_report_filename(
                model_name, feature_type, n_categories
            )
            report_file = results_path / filename
            report_df.to_csv(report_file)
            
            logger.info(f"Classification report saved: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Error generating classification report for {model_name}: {e}")
            return None
    
    def print_model_metrics(self, results, model_name, n_categories, feature_type, training_time, model_category):
        """Print standardized model metrics"""
        print(f"\n{'='*60}")
        print(f"{model_category} MODEL EVALUATION: {model_name}")
        print(f"{'='*60}")
        print(f"Categories: {n_categories} | Feature Type: {feature_type.upper()}")
        print(f"Training Time: {training_time:.2f}s | Inference Time: {results.get('inference_time', 0):.4f}s")
        print(f"{'-'*60}")
        print(f"Top-1 Accuracy: {results.get('top1_accuracy', results.get('accuracy', 0)):.4f}")
        print(f"Top-3 Accuracy: {results.get('top3_accuracy', 0):.4f}")
        print(f"Top-5 Accuracy: {results.get('top5_accuracy', 0):.4f}")
        print(f"Macro F1:      {results.get('macro_f1', 0):.4f}")
        print(f"Micro F1:      {results.get('micro_f1', 0):.4f}")
        print(f"{'='*60}")
    
    def save_model_performance_data(self, results, model_name, n_categories, feature_type, model_type):
        """Save model performance data to final results"""
        try:
            # Initialize category if not exists
            if n_categories not in self.final_results:
                self.final_results[n_categories] = {}
            
            # Create standardized key
            clean_model_name = FileNamingStandard.standardize_model_name(model_name)
            result_key = f"{clean_model_name}_{feature_type}"
            
            # Store results
            self.final_results[n_categories][result_key] = results
            
            # Save to pickle file for overall analysis
            self._save_final_results_pickle(model_type)
            
            logger.info(f"Performance data saved for {model_name} ({feature_type})")
            
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def _save_final_results_pickle(self, model_type):
        """Save final results as pickle file for overall analysis"""
        try:
            # Determine comparisons path
            if model_type == 'ml':
                comparisons_path = RESULTS_CONFIG['ml_comparisons_path']
            elif model_type == 'dl':
                comparisons_path = RESULTS_CONFIG['dl_comparisons_path']
            elif model_type in ['bert', 'roberta']:
                comparisons_path = RESULTS_CONFIG['bert_comparisons_path']
            elif model_type == 'deepseek':
                comparisons_path = RESULTS_CONFIG['deepseek_comparisons_path']
            elif model_type == 'fusion':  # ✓ ADD THIS
                comparisons_path = RESULTS_CONFIG['fusion_comparisons_path']
            else:
                return  # Skip if unknown type
            
            comparisons_path.mkdir(parents=True, exist_ok=True)
            
            # Save as pickle
            pickle_file = comparisons_path / f"{model_type}_final_results.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(self.final_results, f)
                
            logger.info(f"Final results saved: {pickle_file}")
            
        except Exception as e:
            logger.error(f"Error saving final results pickle: {e}")
    
    def plot_results_comparison(self, results_file_path, charts_dir, model_type):
        """Generate comparison plots with full implementation"""
        try:
            if not results_file_path.exists():
                print(f"No {model_type.upper()} results file found at: {results_file_path}")
                return
            
            # Create charts directory
            charts_dir = Path(charts_dir)
            charts_dir.mkdir(parents=True, exist_ok=True)
            
            # Load results
            with open(results_file_path, "rb") as f:
                final_results = pickle.load(f)
            
            print(f"Generating plots and analysis for {model_type.upper()} results...")
            print(f"Results loaded from: {results_file_path}")
            print(f"Charts will be saved to: {charts_dir}")
            
            model_metrics = {}

            # Parse results and organize by model and feature type
            for n, results in final_results.items():
                for entry in results:
                    if isinstance(entry, dict):
                        model = entry.get('model', 'Unknown')
                        feature_type = entry.get('feature_type', 'unknown')
                        
                        if model not in model_metrics:
                            model_metrics[model] = {}
                        
                        if feature_type not in model_metrics[model]:
                            model_metrics[model][feature_type] = {
                                'n': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
                                'top1_accuracy': [], 'top3_accuracy': [], 'top5_accuracy': [],
                                'training_time': [], 'inference_time': []
                            }
                        
                        model_metrics[model][feature_type]['n'].append(entry.get('n_categories', n))
                        model_metrics[model][feature_type]['accuracy'].append(entry.get('accuracy', 0))
                        model_metrics[model][feature_type]['precision'].append(entry.get('precision', entry.get('macro_precision', 0)))
                        model_metrics[model][feature_type]['recall'].append(entry.get('recall', entry.get('macro_recall', 0)))
                        model_metrics[model][feature_type]['f1_score'].append(entry.get('f1_score', entry.get('macro_f1', 0)))
                        model_metrics[model][feature_type]['top1_accuracy'].append(entry.get('top1_accuracy', entry.get('accuracy', 0)))
                        model_metrics[model][feature_type]['top3_accuracy'].append(entry.get('top3_accuracy', 0))
                        model_metrics[model][feature_type]['top5_accuracy'].append(entry.get('top5_accuracy', 0))
                        model_metrics[model][feature_type]['training_time'].append(entry.get('training_time', 0))
                        model_metrics[model][feature_type]['inference_time'].append(entry.get('inference_time', 0))
            
            # Generate line plots
            self._generate_line_plots(model_metrics, charts_dir, model_type)
            
            # Generate bar plots for each category
            self._generate_bar_plots(final_results, charts_dir, model_type)
            
            # Generate summary statistics
            self._generate_summary_statistics(final_results, charts_dir, model_type)
                
        except Exception as e:
            logger.error(f"Error generating {model_type} plots: {e}")
    
    def _generate_line_plots(self, model_metrics, charts_dir, model_type):
        """Generate line plots for performance metrics"""
        def plot_metric(metric_name, ylabel=None):
            plt.figure(figsize=(12, 6))
            
            for model, features in model_metrics.items():
                for feature_type, data in features.items():
                    label = f"{model} ({feature_type.upper()})"
                    plt.plot(data['n'], data[metric_name], marker='o', label=label, linewidth=2)
            
            plt.title(f'{ylabel or metric_name.replace("_", " ").title()} vs Number of Web Service Categories ({model_type.upper()} Models)')
            plt.xlabel('Number of Web Service Categories')
            plt.ylabel(ylabel or metric_name.replace("_", " ").title())
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plot_path = charts_dir / f"{model_type.upper()}_Model_Performance_{metric_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"{model_type.upper()} line plot saved: {plot_path}")
            plt.close()

        # Generate line plots for all metrics
        print(f"\nGenerating {model_type.upper()} line plots...")
        metrics_config = {
            'accuracy': 'Accuracy',
            'precision': 'Precision (Macro)',
            'recall': 'Recall (Macro)',
            'f1_score': 'F1-Score (Macro)',
            'top1_accuracy': 'Top-1 Accuracy',
            'top3_accuracy': 'Top-3 Accuracy', 
            'top5_accuracy': 'Top-5 Accuracy'
        }
        
        if model_type.lower() == "dl":
            metrics_config.update({
                'training_time': 'Training Time (seconds)',
                'inference_time': 'Inference Time (seconds)'
            })
        
        for metric, ylabel in metrics_config.items():
            plot_metric(metric, ylabel)
        
        # Combined top-K accuracy plot
        print(f"\nGenerating combined {model_type.upper()} top-K accuracy plot...")
        plt.figure(figsize=(14, 8))
        
        for model, features in model_metrics.items():
            for feature_type, data in features.items():
                label_base = f"{model} ({feature_type.upper()})"
                plt.plot(data['n'], data['top1_accuracy'], marker='o', label=f"{label_base} - Top-1", linewidth=2)
                plt.plot(data['n'], data['top3_accuracy'], marker='s', label=f"{label_base} - Top-3", linewidth=2, linestyle='--')
                plt.plot(data['n'], data['top5_accuracy'], marker='^', label=f"{label_base} - Top-5", linewidth=2, linestyle=':')
        
        plt.title(f'{model_type.upper()} Models: Top-K Accuracy Comparison')
        plt.xlabel('Number of Web Service Categories')
        plt.ylabel('Top-K Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plot_path = charts_dir / f"{model_type.upper()}_Model_Performance_topk_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Combined {model_type.upper()} top-K plot saved: {plot_path}")
        plt.close()
    
    def _generate_bar_plots(self, final_results, charts_dir, model_type):
        """Generate bar plots for each category size"""
        print(f"\nGenerating enhanced {model_type.upper()} bar plots for each category size...")
        
        for n in CATEGORY_SIZES:
            if n not in final_results:
                print(f"Skipping n={n} (no {model_type.upper()} results found)")
                continue

            # Handle different result structures
            combined_results = []
            if isinstance(final_results[n], list):
                combined_results = final_results[n]
            elif isinstance(final_results[n], dict):
                for key, value in final_results[n].items():
                    if isinstance(value, dict):
                        # Add model name if missing
                        if 'model' not in value:
                            value['model'] = key
                        combined_results.append(value)
            
            if not combined_results:
                continue
                
            df_combined = pd.DataFrame(combined_results)

            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'top1_accuracy', 'top3_accuracy', 'top5_accuracy']
            available_metrics = [col for col in metrics_to_plot if col in df_combined.columns]
            
            # Handle missing model column
            if 'model' not in df_combined.columns:
                df_combined['model'] = [f'Model_{i}' for i in range(len(df_combined))]
            
            df_plot_combined = df_combined[['model'] + available_metrics]
            df_plot_combined.set_index('model', inplace=True)

            # Main performance plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Standard metrics
            standard_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            available_standard = [m for m in standard_metrics if m in df_plot_combined.columns]
            if available_standard:
                df_plot_combined[available_standard].plot(kind='bar', ax=ax1, width=0.8)
                ax1.set_title(f'{model_type.upper()} Standard Performance Metrics - Top {n} Categories', fontsize=14)
                ax1.set_ylabel('Score')
                ax1.set_ylim(0, 1.0)
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(axis='y', alpha=0.3)
                ax1.legend(title='Metric')
            
            # Top-K accuracy metrics
            topk_metrics = ['top1_accuracy', 'top3_accuracy', 'top5_accuracy']
            available_topk = [col for col in topk_metrics if col in df_plot_combined.columns]
            if available_topk:
                df_plot_combined[available_topk].plot(kind='bar', ax=ax2, width=0.8)
                ax2.set_title(f'{model_type.upper()} Top-K Accuracy Metrics - Top {n} Categories', fontsize=14)
                ax2.set_ylabel('Top-K Accuracy')
                ax2.set_ylim(0, 1.0)
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(axis='y', alpha=0.3)
                ax2.legend(title='Top-K Metric')
            
            plt.tight_layout()
            
            plot_path = charts_dir / f"{model_type.upper()}_Model_Performance_enhanced_top_{n}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Enhanced {model_type.upper()} bar plot saved: {plot_path}")
            plt.close()
    
    def _generate_summary_statistics(self, final_results, charts_dir, model_type):
        """Generate summary statistics table"""
        print(f"\nGenerating {model_type.upper()} summary statistics...")
        
        # Create summary table
        summary_data = []
        for n in CATEGORY_SIZES:
            if n in final_results:
                # Handle different result structures
                results_list = []
                if isinstance(final_results[n], list):
                    results_list = final_results[n]
                elif isinstance(final_results[n], dict):
                    for key, value in final_results[n].items():
                        if isinstance(value, dict):
                            if 'model' not in value:
                                value['model'] = key
                            results_list.append(value)
                
                for entry in results_list:
                    if isinstance(entry, dict):
                        summary_entry = {
                            'Categories': n,
                            'Model': entry.get('model', 'Unknown'),
                            'Feature': entry.get('feature_type', 'unknown'),
                            'Accuracy': entry.get('accuracy', 0),
                            'F1-Score': entry.get('f1_score', entry.get('macro_f1', 0)),
                            'Top-1': entry.get('top1_accuracy', entry.get('accuracy', 0)),
                            'Top-3': entry.get('top3_accuracy', 0),
                            'Top-5': entry.get('top5_accuracy', 0)
                        }
                        
                        if model_type.lower() == "dl":
                            summary_entry.update({
                                'Training Time': entry.get('training_time', 0),
                                'Inference Time': entry.get('inference_time', 0)
                            })
                        
                        summary_data.append(summary_entry)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.round(4)
            
            # Save summary table
            summary_path = charts_dir / f"{model_type.upper()}_Model_Performance_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"{model_type.upper()} summary table saved: {summary_path}")
            
            # Display best performing models
            print(f"\nTop performing {model_type.upper()} models by metric:")
            for metric in ['Accuracy', 'F1-Score', 'Top-1', 'Top-3', 'Top-5']:
                if metric in summary_df.columns and len(summary_df) > 0:
                    best = summary_df.loc[summary_df[metric].idxmax()]
                    print(f"  {metric}: {best['Model']} ({best['Feature']}) on {best['Categories']} categories = {best[metric]:.4f}")
            
            # Best model overall
            if len(summary_df) > 0 and 'Top-1' in summary_df.columns:
                best_overall = summary_df.loc[summary_df['Top-1'].idxmax()]
                print(f"\nBest Overall {model_type.upper()} Model:")
                print(f"  {best_overall['Model']} ({best_overall['Feature']}) on {best_overall['Categories']} categories")
                print(f"  Top-1 Accuracy: {best_overall['Top-1']:.4f}")
                print(f"  F1-Score: {best_overall['F1-Score']:.4f}")
                if model_type.lower() == "dl" and 'Training Time' in summary_df.columns:
                    print(f"  Training Time: {best_overall['Training Time']:.2f}s")
    
    def generate_radar_plots(self, model_type, show_plots=False):
        """Generate radar plots for model performance across categories"""
        from math import pi
        
        # Model naming patterns for file reading
        NAMING_PATTERNS = {
            "logistic_regression": "LogisticRegression",
            "random_forest": "RandomForest",        
            "xgboost": "XGBoost",
            "bilstm": "BiLSTM"
        }
        
        # Get model configuration based on type
        if model_type.lower() == "ml":
            try:
                from src.config import ML_CONFIG
                models = ML_CONFIG.get("models", ["LogisticRegression", "RandomForest", "XGBoost"])
                results_paths = RESULTS_CONFIG["ml_category_paths"]
                save_dir = RESULTS_CONFIG["ml_comparisons_path"]
                title_prefix = "ML Models"
                feature_types = ["tfidf", "sbert"]
            except:
                models = ["LogisticRegression", "RandomForest", "XGBoost"]
                results_paths = RESULTS_CONFIG["ml_category_paths"]
                save_dir = RESULTS_CONFIG["ml_comparisons_path"]
                title_prefix = "ML Models"
                feature_types = ["tfidf", "sbert"]
        elif model_type.lower() == "dl":
            try:
                from src.config import DL_CONFIG
                models = DL_CONFIG.get("models", ["BiLSTM"]) 
                results_paths = RESULTS_CONFIG["dl_category_paths"]
                save_dir = RESULTS_CONFIG["dl_comparisons_path"]
                title_prefix = "DL Models"
                feature_types = DL_CONFIG.get("feature_types", ["tfidf", "sbert"])
            except:
                models = ["BiLSTM"]
                results_paths = RESULTS_CONFIG["dl_category_paths"]
                save_dir = RESULTS_CONFIG["dl_comparisons_path"]
                title_prefix = "DL Models"
                feature_types = ["tfidf", "sbert"]
        elif model_type.lower() == "bert":
            models = ["RoBERTa_Base", "RoBERTa_Large"]
            results_paths = RESULTS_CONFIG["bert_category_paths"]
            save_dir = RESULTS_CONFIG["bert_comparisons_path"]
            title_prefix = "BERT Models"
            feature_types = ["raw_text"]
        elif model_type.lower() == "deepseek":
            models = ["DeepSeek_7B_Base"]
            results_paths = RESULTS_CONFIG["deepseek_category_paths"]
            save_dir = RESULTS_CONFIG["deepseek_comparisons_path"]
            title_prefix = "DeepSeek Models"
            feature_types = ["raw_text"]
        else:
            logger.warning(f"Unknown model type: {model_type}")
            return
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics to plot
        metrics = ["precision", "recall", "f1-score", "accuracy"]
        
        print(f"\nGenerating {model_type.upper()} radar plots...")
        
        for num_cat in CATEGORY_SIZES:
            print(f"Processing radar plots for {num_cat} categories...")
            
            # Load classification reports for this category size
            data = self._load_radar_data(models, feature_types, results_paths[num_cat], num_cat, NAMING_PATTERNS)
            
            if not data:
                print(f"No data found for {num_cat} categories, skipping radar plots")
                continue
            
            # Generate radar plot for each metric
            for metric in metrics:
                self._plot_radar_chart(data, metric, num_cat, title_prefix, save_dir, model_type, show_plots)
        
        print(f"Completed {model_type.upper()} radar plot generation")
    
    def _load_radar_data(self, models, feature_types, category_path, num_cat, naming_patterns):
        """Load classification report data for radar plots"""
        data = {}
        
        for model in models:
            for feature in feature_types:
                # Convert model name using naming patterns
                model_display_name = naming_patterns.get(model, model)
                filename = FileNamingStandard.generate_classification_report_filename(
                    model_display_name, feature, num_cat
                )
                file_path = category_path / filename

                if not file_path.exists():
                    logger.warning(f"Missing radar data file: {file_path}")
                    continue

                try:
                    # Read classification report CSV
                    df = pd.read_csv(file_path)
                    
                    # Filter to only category rows (exclude macro/micro/weighted avg rows)
                    if 'category_name' in df.columns:
                        # Filter out summary rows
                        category_rows = df[~df['category_name'].isin(['macro avg', 'micro avg', 'weighted avg'])]
                        category_rows = category_rows[~category_rows['category_name'].isna()]
                        
                        # Take only first num_cat rows and set index
                        category_rows = category_rows.head(num_cat).set_index("category_name")
                        data[f"{model}_{feature}"] = category_rows
                        
                    else:
                        logger.warning(f"No 'category_name' column found in {file_path}")
                        
                except Exception as e:
                    logger.error(f"Error reading radar data from {file_path}: {e}")

        return data
    
    def _plot_radar_chart(self, data, metric, num_cat, title_prefix, save_dir, model_type, show_plots):
        """Generate and save radar chart for given metric"""
        if not data:
            return

        # Get category labels from first dataset
        first_key = list(data.keys())[0]
        if first_key not in data or data[first_key].empty:
            return
            
        labels = data[first_key].index.tolist()
        num_labels = len(labels)
        
        if num_labels == 0:
            logger.warning(f"No labels found for radar plot with {num_cat} categories")
            return
        
        # Calculate angles for radar chart
        angles = [n / float(num_labels) * 2 * pi for n in range(num_labels)]
        angles += angles[:1]  # Complete the circle

        # Create figure with appropriate size
        figsize = (8, 8) if num_cat < 40 else (14, 14)
        plt.figure(figsize=figsize)
        ax = plt.subplot(111, polar=True)

        # Plot each model-feature combination
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        
        for i, (model_name, df) in enumerate(data.items()):
            if metric not in df.columns:
                logger.warning(f"Metric '{metric}' not found in data for {model_name}")
                continue
                
            # Get metric values, handling NaN values
            metric_values = df[metric].fillna(0).tolist()
            
            # Ensure we have the right number of values
            if len(metric_values) != num_labels:
                logger.warning(f"Metric values length mismatch for {model_name}: expected {num_labels}, got {len(metric_values)}")
                continue
            
            # Complete the circle for plotting
            values = metric_values + metric_values[:1]
            
            # Create readable label
            display_name = model_name.replace('_', ' ').title()
            
            # Plot the radar line
            ax.plot(angles, values, 'o-', linewidth=2, label=display_name, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        # Customize the plot
        ax.set_xticks(angles[:-1])
        
        # Adjust label size and rotation based on number of categories
        fontsize = 10 if num_cat < 20 else 8 if num_cat < 40 else 6
        
        # Truncate long labels for readability
        display_labels = [lbl[:15] + "..." if len(lbl) > 18 else lbl for lbl in labels]
        ax.set_xticklabels(display_labels, fontsize=fontsize)
        
        # Set title and limits
        ax.set_title(f"{title_prefix} - {metric.replace('-', ' ').title()} Performance\n(Top {num_cat} Categories)",
                     size=16 if num_cat < 40 else 14, weight="bold", pad=20)
        ax.set_ylim(0, 1)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Position legend appropriately
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.tight_layout()

        # Save the plot
        filename = f"{model_type.upper()}_radar_{metric.replace('-', '_')}_top_{num_cat}_categories.png"
        filepath = save_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Radar plot saved: {filepath}")

        if show_plots:
            plt.show()
        plt.close()