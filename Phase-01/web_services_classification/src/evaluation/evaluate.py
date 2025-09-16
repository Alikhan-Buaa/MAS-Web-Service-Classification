"""
Common Evaluation Module for ML and DL Models
Contains shared evaluation functionality, visualization, and result analysis
"""

import pandas as pd
import numpy as np
import logging
import json
import yaml
import time
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from math import pi

# Import configuration
from src.config import (
    CATEGORY_SIZES, RESULTS_CONFIG, SAVED_MODELS_CONFIG, PREPROCESSING_CONFIG
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Common evaluation class for both ML and DL models"""
    
    def __init__(self):
        self.final_results = {}
    
    def calculate_top_k_accuracy(self, y_true, y_proba, k=5):
        """Calculate Top-K accuracy for both ML and DL models"""
        try:
            if k == 1:
                if hasattr(y_true, 'ndim') and y_true.ndim > 1:
                    # For DL models with one-hot encoded labels
                    y_pred = np.argmax(y_proba, axis=1)
                    y_true_labels = np.argmax(y_true, axis=1)
                    return accuracy_score(y_true_labels, y_pred)
                else:
                    # For ML models with label indices
                    y_pred = np.argmax(y_proba, axis=1)
                    return accuracy_score(y_true, y_pred)
            
            # Handle one-hot encoded labels for DL
            if hasattr(y_true, 'ndim') and y_true.ndim > 1:
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
    
    def generate_confusion_heatmap(self, cm, class_labels, model_name, n_categories, feature_type, model_type="ml"):
        """Generate confusion matrix heatmap with dynamic sizing"""
        n_classes = len(class_labels)
        
        # Dynamic figure size based on number of classes
        figsize = (12, 10)
        if n_classes >= 40:
            figsize = (18, 18)
        elif n_classes > 20:
            figsize = (14, 12)
        
        # Adjust annotation font size based on number of classes
        annot_fontsize = 8 if n_classes <= 20 else 6 if n_classes <= 40 else 4
        
        plt.figure(figsize=figsize)
        
        # Shorten labels if too long
        display_labels = [lbl[:12] + "..." if len(lbl) > 15 else lbl for lbl in class_labels]
        
        # Create heatmap
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=display_labels, yticklabels=display_labels,
            annot_kws={"size": annot_fontsize}
        )
        
        plt.title(f'Confusion Matrix: {model_name} ({feature_type.upper()}, Top-{n_categories})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save to appropriate directory based on model type
        if model_type.lower() == "dl":
            cm_dir = RESULTS_CONFIG["dl_category_paths"][n_categories]
        else:
            cm_dir = RESULTS_CONFIG["ml_category_paths"][n_categories]
        
        cm_dir.mkdir(parents=True, exist_ok=True)
        save_path = cm_dir / f"{model_name}_{feature_type}_top_{n_categories}_categories_confusion_matrix.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return save_path
    
    def generate_classification_report_csv(self, y_true, y_pred, class_labels, model_name, n_categories, feature_type, model_type="ml"):
        """Generate classification report with accuracy per class"""
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        df = pd.DataFrame(report_dict).transpose()

        cm = confusion_matrix(y_true, y_pred)
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
        df["accuracy"] = np.concatenate([per_class_acc, [None] * (len(df) - len(per_class_acc))])
        
        # This is the key line that adds the category_name column
        df["category_name"] = df.index.map(
            lambda x: class_labels[int(x)] if x.isdigit() and int(x) < len(class_labels) else x
        )

        df.reset_index(inplace=True)
        df.rename(columns={"index": "Class"}, inplace=True)

        # Save to appropriate directory based on model type
        if model_type.lower() == "dl":
            reports_dir = RESULTS_CONFIG["dl_category_paths"][n_categories]
        else:
            reports_dir = RESULTS_CONFIG["ml_category_paths"][n_categories]
        
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir / f"{model_name}_{feature_type}_top_{n_categories}_categories_classification_report.csv"
        df.to_csv(out_path, index=False)
        return out_path
    
    def print_model_metrics(self, results, model_name, n_categories, feature_type, training_time, model_type="ML"):
        """Print model metrics to console in a formatted way"""
        print(f"\n{'='*60}")
        print(f"{model_type.upper()} MODEL PERFORMANCE SUMMARY")
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
    
    def save_model_performance_data(self, results, model_name, n_categories, feature_type, model_type="ml"):
        """Save model performance data in the format needed for plotting"""
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
        
        # Store in final_results structure
        if n_categories not in self.final_results:
            self.final_results[n_categories] = []
        
        self.final_results[n_categories].append(model_entry)
        
        # Save after each model
        self.save_final_results(model_type)
        
        return model_entry
    
    def save_final_results(self, model_type="ml"):
        """Save final results to pickle file for plotting"""
        if model_type.lower() == "dl":
            models_path = SAVED_MODELS_CONFIG["dl_models_path"]
            comparisons_path = RESULTS_CONFIG["dl_comparisons_path"]
            filename = "dl_final_results.pkl"
        else:
            models_path = SAVED_MODELS_CONFIG["ml_models_path"]
            comparisons_path = RESULTS_CONFIG["ml_comparisons_path"]
            filename = "ml_final_results.pkl"
        
        # Save to models directory
        models_path.mkdir(parents=True, exist_ok=True)
        results_file = models_path / filename
        with open(results_file, "wb") as f:
            pickle.dump(self.final_results, f)
        logger.info(f"{model_type.upper()} final results saved to {results_file}")

        # Save to comparisons directory
        comparisons_path.mkdir(parents=True, exist_ok=True)
        results_file = comparisons_path / filename
        with open(results_file, "wb") as f:
            pickle.dump(self.final_results, f)
        logger.info(f"{model_type.upper()} final results saved to {results_file}")
    
    def load_class_labels(self, n_categories):
        """Load class labels from YAML file"""
        labels_file = Path(PREPROCESSING_CONFIG["labels"].format(n=n_categories))
        try:
            with open(labels_file, "r") as f:
                class_labels = yaml.safe_load(f).get("categories", [])
        except FileNotFoundError:
            class_labels = [f'Cat_{i}' for i in range(n_categories)]
        return class_labels
    
    def plot_results_comparison(self, results_file_path, charts_dir, model_type="ml"):
        """
        Generate comprehensive plots and analysis
        
        Args:
            results_file_path (Path): Path to the final_results.pkl file
            charts_dir (Path): Directory to save charts
            model_type (str): "ml" or "dl"
        """
        
        if not results_file_path.exists():
            print(f"No {model_type.upper()} results file found at: {results_file_path}")
            return
        
        # Create charts directory
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
                model = entry['model']
                feature_type = entry['feature_type']
                
                if model not in model_metrics:
                    model_metrics[model] = {}
                
                if feature_type not in model_metrics[model]:
                    model_metrics[model][feature_type] = {
                        'n': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
                        'top1_accuracy': [], 'top3_accuracy': [], 'top5_accuracy': [],
                        'training_time': [], 'inference_time': []
                    }
                
                model_metrics[model][feature_type]['n'].append(entry['n_categories'])
                model_metrics[model][feature_type]['accuracy'].append(entry['accuracy'])
                model_metrics[model][feature_type]['precision'].append(entry['precision'])
                model_metrics[model][feature_type]['recall'].append(entry['recall'])
                model_metrics[model][feature_type]['f1_score'].append(entry['f1_score'])
                model_metrics[model][feature_type]['top1_accuracy'].append(entry.get('top1_accuracy', entry['accuracy']))
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

            combined_results = final_results[n]
            df_combined = pd.DataFrame(combined_results)

            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'top1_accuracy', 'top3_accuracy', 'top5_accuracy']
            available_metrics = [col for col in metrics_to_plot if col in df_combined.columns]
            
            df_plot_combined = df_combined[['model'] + available_metrics]
            df_plot_combined.set_index('model', inplace=True)

            # Main performance plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Standard metrics
            standard_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            df_plot_combined[standard_metrics].plot(kind='bar', ax=ax1, width=0.8)
            ax1.set_title(f'{model_type.upper()} Standard Performance Metrics - Top {n} Categories', fontsize=14)
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
                ax2.set_title(f'{model_type.upper()} Top-K Accuracy Metrics - Top {n} Categories', fontsize=14)
                ax2.set_ylabel('Top-K Accuracy')
                ax2.set_ylim(0.3, 1.0)
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
                for entry in final_results[n]:
                    summary_entry = {
                        'Categories': n,
                        'Model': entry['model'],
                        'Feature': entry['feature_type'],
                        'Accuracy': entry['accuracy'],
                        'F1-Score': entry['f1_score'],
                        'Top-1': entry.get('top1_accuracy', entry['accuracy']),
                        'Top-3': entry.get('top3_accuracy', 0),
                        'Top-5': entry.get('top5_accuracy', 0)
                    }
                    
                    if model_type.lower() == "dl":
                        summary_entry.update({
                            'Training Time': entry.get('training_time', 0),
                            'Inference Time': entry.get('inference_time', 0)
                        })
                    
                    summary_data.append(summary_entry)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.round(4)
        
        # Save summary table
        summary_path = charts_dir / f"{model_type.upper()}_Model_Performance_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"{model_type.upper()} summary table saved: {summary_path}")
        
        # Display best performing models
        print(f"\nTop performing {model_type.upper()} models by metric:")
        for metric in ['Accuracy', 'F1-Score', 'Top-1', 'Top-3', 'Top-5']:
            if metric in summary_df.columns:
                best = summary_df.loc[summary_df[metric].idxmax()]
                print(f"  {metric}: {best['Model']} ({best['Feature']}) on {best['Categories']} categories = {best[metric]:.4f}")
        
        # Best model overall
        if len(summary_df) > 0:
            best_overall = summary_df.loc[summary_df['Top-1'].idxmax()]
            print(f"\nBest Overall {model_type.upper()} Model:")
            print(f"  {best_overall['Model']} ({best_overall['Feature']}) on {best_overall['Categories']} categories")
            print(f"  Top-1 Accuracy: {best_overall['Top-1']:.4f}")
            print(f"  F1-Score: {best_overall['F1-Score']:.4f}")
            if model_type.lower() == "dl" and 'Training Time' in summary_df.columns:
                print(f"  Training Time: {best_overall['Training Time']:.2f}s")
    
    def generate_radar_plots(self, model_type="ml", show_plots=False):
        """
        Generate radar plots for model performance across categories
        
        Args:
            model_type (str): "ml" or "dl"
            show_plots (bool): If True, also show interactive plots
        """
        from math import pi
        
        # Model naming patterns for file reading
        NAMING_PATTERNS = {
            # snake_case (config) → PascalCase (model name)
            "logistic_regression": "LogisticRegression",
            "random_forest": "RandomForest",        
            "xgboost": "XGBoost",
            "bilstm": "BiLSTM"
        }
        
        # Get model configuration based on type
        if model_type.lower() == "ml":
            from src.config import ML_CONFIG
            models = ML_CONFIG["models"]
            results_paths = RESULTS_CONFIG["ml_category_paths"]
            save_dir = RESULTS_CONFIG["ml_comparisons_path"]
            title_prefix = "ML Models"
            feature_types = ["tfidf", "sbert"]
        else:
            from src.config import DL_CONFIG
            models = DL_CONFIG["models"] 
            results_paths = RESULTS_CONFIG["dl_category_paths"]
            save_dir = RESULTS_CONFIG["dl_comparisons_path"]
            title_prefix = "DL Models"
            feature_types = DL_CONFIG.get("feature_types", ["tfidf", "sbert"])
        
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
                file_name = f"{model_display_name}_{feature}_top_{num_cat}_categories_classification_report.csv"
                file_path = category_path / file_name

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
    
    def plot_results_with_radar(self, results_file_path, charts_dir, model_type="ml", show_plots=False):
        """
        Generate comprehensive plots including radar charts
        
        Args:
            results_file_path (Path): Path to the final_results.pkl file
            charts_dir (Path): Directory to save charts
            model_type (str): "ml" or "dl" 
            show_plots (bool): Whether to show interactive plots
        """
        # Generate standard comparison plots
        self.plot_results_comparison(results_file_path, charts_dir, model_type)
        
        # Generate radar plots
        self.generate_radar_plots(model_type, show_plots)