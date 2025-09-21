"""
Overall Performance Comparison Module
Combines ML and DL model results for comprehensive analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pickle
from pathlib import Path
from math import pi

from src.config import (
    CATEGORY_SIZES, RESULTS_CONFIG, ML_CONFIG, DL_CONFIG, PREPROCESSING_CONFIG
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OverallPerformanceAnalyzer:
    """Comprehensive analyzer for combined ML and DL model performance"""
    
    def __init__(self):
        # Create overall results directory
        self.overall_dir = RESULTS_CONFIG.get("results_path", Path("results")) / "overall"
        self.overall_dir.mkdir(parents=True, exist_ok=True)
        
        # Model naming patterns
        self.naming_patterns = {
            "logistic_regression": "LogisticRegression",
            "random_forest": "RandomForest",
            "xgboost": "XGBoost", 
            "bilstm": "BiLSTM"
        }
        
    def load_all_results(self):
        """Load both ML and DL results"""
        ml_results_file = RESULTS_CONFIG["ml_comparisons_path"] / "ml_final_results.pkl"
        dl_results_file = RESULTS_CONFIG["dl_comparisons_path"] / "dl_final_results.pkl"
        
        ml_data, dl_data = None, None
        
        # Load ML results
        try:
            if ml_results_file.exists():
                with open(ml_results_file, "rb") as f:
                    ml_data = pickle.load(f)
                logger.info(f"Loaded ML results: {len(ml_data)} category sizes")
            else:
                logger.warning(f"ML results file not found: {ml_results_file}")
        except Exception as e:
            logger.error(f"Error loading ML results: {e}")
        
        # Load DL results  
        try:
            if dl_results_file.exists():
                with open(dl_results_file, "rb") as f:
                    dl_data = pickle.load(f)
                logger.info(f"Loaded DL results: {len(dl_data)} category sizes")
            else:
                logger.warning(f"DL results file not found: {dl_results_file}")
        except Exception as e:
            logger.error(f"Error loading DL results: {e}")
            
        return ml_data, dl_data
    
    def combine_results_for_plotting(self, ml_data, dl_data):
        """Combine ML and DL data into unified structure for plotting"""
        combined_metrics = {}
        
        # Process ML data
        if ml_data:
            for n_categories, results in ml_data.items():
                for entry in results:
                    model_key = f"{entry['model']} (ML)"
                    feature_type = entry['feature_type']
                    
                    if model_key not in combined_metrics:
                        combined_metrics[model_key] = {}
                    if feature_type not in combined_metrics[model_key]:
                        combined_metrics[model_key][feature_type] = {
                            'n': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
                            'top1_accuracy': [], 'top3_accuracy': [], 'top5_accuracy': [],
                            'training_time': [], 'inference_time': [], 'model_type': 'ML'
                        }
                    
                    combined_metrics[model_key][feature_type]['n'].append(entry['n_categories'])
                    combined_metrics[model_key][feature_type]['accuracy'].append(entry['accuracy'])
                    combined_metrics[model_key][feature_type]['precision'].append(entry['precision'])
                    combined_metrics[model_key][feature_type]['recall'].append(entry['recall'])
                    combined_metrics[model_key][feature_type]['f1_score'].append(entry['f1_score'])
                    combined_metrics[model_key][feature_type]['top1_accuracy'].append(entry.get('top1_accuracy', entry['accuracy']))
                    combined_metrics[model_key][feature_type]['top3_accuracy'].append(entry.get('top3_accuracy', 0))
                    combined_metrics[model_key][feature_type]['top5_accuracy'].append(entry.get('top5_accuracy', 0))
                    combined_metrics[model_key][feature_type]['training_time'].append(entry.get('training_time', 0))
                    combined_metrics[model_key][feature_type]['inference_time'].append(entry.get('inference_time', 0))
        
        # Process DL data
        if dl_data:
            for n_categories, results in dl_data.items():
                for entry in results:
                    model_key = f"{entry['model']} (DL)"
                    feature_type = entry['feature_type']
                    
                    if model_key not in combined_metrics:
                        combined_metrics[model_key] = {}
                    if feature_type not in combined_metrics[model_key]:
                        combined_metrics[model_key][feature_type] = {
                            'n': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
                            'top1_accuracy': [], 'top3_accuracy': [], 'top5_accuracy': [],
                            'training_time': [], 'inference_time': [], 'model_type': 'DL'
                        }
                    
                    combined_metrics[model_key][feature_type]['n'].append(entry['n_categories'])
                    combined_metrics[model_key][feature_type]['accuracy'].append(entry['accuracy'])
                    combined_metrics[model_key][feature_type]['precision'].append(entry['precision'])
                    combined_metrics[model_key][feature_type]['recall'].append(entry['recall'])
                    combined_metrics[model_key][feature_type]['f1_score'].append(entry['f1_score'])
                    combined_metrics[model_key][feature_type]['top1_accuracy'].append(entry.get('top1_accuracy', entry['accuracy']))
                    combined_metrics[model_key][feature_type]['top3_accuracy'].append(entry.get('top3_accuracy', 0))
                    combined_metrics[model_key][feature_type]['top5_accuracy'].append(entry.get('top5_accuracy', 0))
                    combined_metrics[model_key][feature_type]['training_time'].append(entry.get('training_time', 0))
                    combined_metrics[model_key][feature_type]['inference_time'].append(entry.get('inference_time', 0))
        
        return combined_metrics
    
    def generate_combined_line_plots(self, combined_metrics):
        """Generate line plots comparing ML and DL models"""
        print("\nGenerating combined ML/DL line plots...")
        
        # Define colors for ML vs DL
        ml_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue tones for ML
        dl_colors = ['#d62728', '#9467bd', '#8c564b']  # Red/purple tones for DL
        
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
            plt.figure(figsize=(14, 8))
            
            ml_idx, dl_idx = 0, 0
            
            for model, features in combined_metrics.items():
                for feature_type, data in features.items():
                    if len(data['n']) == 0:
                        continue
                        
                    label = f"{model} ({feature_type.upper()})"
                    
                    # Choose color based on model type
                    if data['model_type'] == 'ML':
                        color = ml_colors[ml_idx % len(ml_colors)]
                        linestyle = '-'
                        ml_idx += 1
                    else:
                        color = dl_colors[dl_idx % len(dl_colors)]
                        linestyle = '--'
                        dl_idx += 1
                    
                    plt.plot(data['n'], data[metric], marker='o', label=label, 
                            linewidth=2, color=color, linestyle=linestyle)
            
            plt.title(f'Overall Model Comparison: {ylabel} vs Number of Categories')
            plt.xlabel('Number of Web Service Categories')
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            plot_path = self.overall_dir / f"Overall_Comparison_{metric}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Combined line plot saved: {plot_path}")
            plt.close()
    
    def generate_combined_topk_plots(self, combined_metrics):
        """Generate combined top-K accuracy comparison plots"""
        print("\nGenerating combined Top-K accuracy plots...")
        
        plt.figure(figsize=(16, 10))
        
        # Define distinct styles for different models
        styles = [
            {'marker': 'o', 'linestyle': '-'},
            {'marker': 's', 'linestyle': '--'},
            {'marker': '^', 'linestyle': '-.'},
            {'marker': 'D', 'linestyle': ':'},
            {'marker': 'v', 'linestyle': '-'},
            {'marker': '<', 'linestyle': '--'},
        ]
        
        style_idx = 0
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        color_idx = 0
        
        for model, features in combined_metrics.items():
            for feature_type, data in features.items():
                if len(data['n']) == 0:
                    continue
                    
                label_base = f"{model} ({feature_type.upper()})"
                style = styles[style_idx % len(styles)]
                base_color = colors[color_idx % len(colors)]
                
                # Plot Top-1, Top-3, Top-5 with variations of the same color
                plt.plot(data['n'], data['top1_accuracy'], 
                        marker=style['marker'], linestyle=style['linestyle'],
                        label=f"{label_base} - Top-1", linewidth=2, 
                        color=base_color, alpha=1.0)
                
                plt.plot(data['n'], data['top3_accuracy'], 
                        marker=style['marker'], linestyle=style['linestyle'],
                        label=f"{label_base} - Top-3", linewidth=2, 
                        color=base_color, alpha=0.7)
                
                plt.plot(data['n'], data['top5_accuracy'], 
                        marker=style['marker'], linestyle=style['linestyle'],
                        label=f"{label_base} - Top-5", linewidth=2, 
                        color=base_color, alpha=0.4)
                
                style_idx += 1
                color_idx += 1
        
        plt.title('Overall Model Comparison: Top-K Accuracy Performance')
        plt.xlabel('Number of Web Service Categories')
        plt.ylabel('Top-K Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plot_path = self.overall_dir / "Overall_TopK_Comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Combined Top-K plot saved: {plot_path}")
        plt.close()
    
    def generate_combined_bar_plots(self, ml_data, dl_data):
        """Generate bar plots comparing best models from ML and DL"""
        print("\nGenerating combined bar plots...")
        
        for n_categories in CATEGORY_SIZES:
            # Collect all results for this category size
            all_results = []
            
            if ml_data and n_categories in ml_data:
                for entry in ml_data[n_categories]:
                    entry_copy = entry.copy()
                    entry_copy['model_type'] = 'ML'
                    all_results.append(entry_copy)
            
            if dl_data and n_categories in dl_data:
                for entry in dl_data[n_categories]:
                    entry_copy = entry.copy()
                    entry_copy['model_type'] = 'DL'
                    all_results.append(entry_copy)
            
            if not all_results:
                continue
                
            df_combined = pd.DataFrame(all_results)
            df_combined['model_display'] = df_combined['model'] + ' (' + df_combined['model_type'] + ')'
            
            # Create bar plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
            
            # Standard metrics
            standard_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            df_plot = df_combined.set_index('model_display')[standard_metrics]
            
            # Color by model type
            colors = ['#1f77b4' if 'ML' in idx else '#d62728' for idx in df_plot.index]
            
            df_plot.plot(kind='bar', ax=ax1, width=0.8, color=colors[:len(df_plot)])
            ax1.set_title(f'Overall Standard Performance Metrics - Top {n_categories} Categories', fontsize=14)
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1.0)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3)
            ax1.legend(title='Metric')
            
            # Top-K accuracy metrics
            topk_metrics = ['top1_accuracy', 'top3_accuracy', 'top5_accuracy']
            available_topk = [col for col in topk_metrics if col in df_combined.columns]
            if available_topk:
                df_plot_topk = df_combined.set_index('model_display')[available_topk]
                df_plot_topk.plot(kind='bar', ax=ax2, width=0.8, color=colors[:len(df_plot_topk)])
                ax2.set_title(f'Overall Top-K Accuracy Metrics - Top {n_categories} Categories', fontsize=14)
                ax2.set_ylabel('Top-K Accuracy')
                ax2.set_ylim(0, 1.0)
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(axis='y', alpha=0.3)
                ax2.legend(title='Top-K Metric')
            
            # Training time comparison (if available)
            if 'training_time' in df_combined.columns:
                df_training = df_combined.set_index('model_display')[['training_time']]
                df_training.plot(kind='bar', ax=ax3, width=0.8, color=colors[:len(df_training)])
                ax3.set_title(f'Training Time Comparison - Top {n_categories} Categories', fontsize=14)
                ax3.set_ylabel('Training Time (seconds)')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(axis='y', alpha=0.3)
                ax3.legend(title='Time')
            
            # Feature type effectiveness
            feature_accuracy = df_combined.groupby(['feature_type', 'model_type'])['accuracy'].mean().unstack(fill_value=0)
            if not feature_accuracy.empty:
                feature_accuracy.plot(kind='bar', ax=ax4, width=0.8)
                ax4.set_title(f'Feature Type Effectiveness - Top {n_categories} Categories', fontsize=14)
                ax4.set_ylabel('Average Accuracy')
                ax4.tick_params(axis='x', rotation=0)
                ax4.grid(axis='y', alpha=0.3)
                ax4.legend(title='Model Type')
            
            plt.tight_layout()
            
            plot_path = self.overall_dir / f"Overall_Bar_Comparison_top_{n_categories}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Combined bar plot saved: {plot_path}")
            plt.close()
    
    def generate_combined_radar_plots(self, ml_data, dl_data):
        """Generate radar plots comparing ML and DL models"""
        print("\nGenerating combined radar plots...")
        
        metrics = ["precision", "recall", "f1-score", "accuracy"]
        
        for n_categories in CATEGORY_SIZES:
            for metric in metrics:
                # Load classification report data for both ML and DL
                radar_data = {}
                
                # Load ML radar data
                ml_category_path = RESULTS_CONFIG["ml_category_paths"][n_categories]
                ml_models = ML_CONFIG.get("models", ["logistic_regression", "random_forest", "xgboost"])
                feature_types = ["tfidf", "sbert"]
                
                for model in ml_models:
                    for feature in feature_types:
                        model_display_name = self.naming_patterns.get(model, model)
                        file_name = f"{model_display_name}_{feature}_top_{n_categories}_categories_classification_report.csv"
                        file_path = ml_category_path / file_name
                        
                        if file_path.exists():
                            try:
                                df = pd.read_csv(file_path)
                                if 'category_name' in df.columns:
                                    category_rows = df[~df['category_name'].isin(['macro avg', 'micro avg', 'weighted avg'])]
                                    category_rows = category_rows[~category_rows['category_name'].isna()]
                                    category_rows = category_rows.head(n_categories).set_index("category_name")
                                    radar_data[f"{model}_{feature} (ML)"] = category_rows
                            except Exception as e:
                                logger.warning(f"Error loading ML radar data from {file_path}: {e}")
                
                # Load DL radar data
                dl_category_path = RESULTS_CONFIG["dl_category_paths"][n_categories]
                dl_models = DL_CONFIG.get("models", ["bilstm"])
                
                for model in dl_models:
                    for feature in feature_types:
                        model_display_name = self.naming_patterns.get(model, model)
                        file_name = f"{model_display_name}_{feature}_top_{n_categories}_categories_classification_report.csv"
                        file_path = dl_category_path / file_name
                        
                        if file_path.exists():
                            try:
                                df = pd.read_csv(file_path)
                                if 'category_name' in df.columns:
                                    category_rows = df[~df['category_name'].isin(['macro avg', 'micro avg', 'weighted avg'])]
                                    category_rows = category_rows[~category_rows['category_name'].isna()]
                                    category_rows = category_rows.head(n_categories).set_index("category_name")
                                    radar_data[f"{model}_{feature} (DL)"] = category_rows
                            except Exception as e:
                                logger.warning(f"Error loading DL radar data from {file_path}: {e}")
                
                # Generate radar plot
                if radar_data:
                    self._plot_combined_radar_chart(radar_data, metric, n_categories)
    
    def _plot_combined_radar_chart(self, data, metric, num_cat):
        """Generate combined radar chart for ML and DL models"""
        if not data:
            return

        # Get category labels from first dataset
        first_key = list(data.keys())[0]
        if first_key not in data or data[first_key].empty:
            return
            
        labels = data[first_key].index.tolist()
        num_labels = len(labels)
        
        if num_labels == 0:
            return
        
        # Calculate angles for radar chart
        angles = [n / float(num_labels) * 2 * pi for n in range(num_labels)]
        angles += angles[:1]

        # Create figure with appropriate size
        figsize = (10, 10) if num_cat < 40 else (16, 16)
        plt.figure(figsize=figsize)
        ax = plt.subplot(111, polar=True)

        # Define colors for ML vs DL
        ml_colors = plt.cm.Blues(np.linspace(0.4, 0.9, 10))
        dl_colors = plt.cm.Reds(np.linspace(0.4, 0.9, 10))
        
        ml_idx, dl_idx = 0, 0

        # Plot each model-feature combination
        for model_name, df in data.items():
            if metric not in df.columns:
                continue
                
            # Get metric values, handling NaN values
            metric_values = df[metric].fillna(0).tolist()
            
            if len(metric_values) != num_labels:
                continue
            
            # Complete the circle for plotting
            values = metric_values + metric_values[:1]
            
            # Choose color and style based on ML/DL
            if "(ML)" in model_name:
                color = ml_colors[ml_idx % len(ml_colors)]
                linestyle = '-'
                ml_idx += 1
            else:
                color = dl_colors[dl_idx % len(dl_colors)]
                linestyle = '--'
                dl_idx += 1
            
            # Create readable label
            display_name = model_name.replace('_', ' ').title()
            
            # Plot the radar line
            ax.plot(angles, values, 'o-', linewidth=2, label=display_name, 
                   color=color, linestyle=linestyle)
            ax.fill(angles, values, alpha=0.1, color=color)

        # Customize the plot
        ax.set_xticks(angles[:-1])
        
        # Adjust label size based on number of categories
        fontsize = 10 if num_cat < 20 else 8 if num_cat < 40 else 6
        display_labels = [lbl[:15] + "..." if len(lbl) > 18 else lbl for lbl in labels]
        ax.set_xticklabels(display_labels, fontsize=fontsize)
        
        ax.set_title(f'Overall Model Comparison: {metric.replace("-", " ").title()} Performance\n(Top {num_cat} Categories)',
                     size=16 if num_cat < 40 else 14, weight="bold", pad=20)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.tight_layout()

        # Save the plot
        filename = f"Overall_radar_{metric.replace('-', '_')}_top_{num_cat}_categories.png"
        filepath = self.overall_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Combined radar plot saved: {filepath}")
        plt.close()
    
    def generate_summary_comparison(self, ml_data, dl_data):
        """Generate summary comparison tables"""
        print("\nGenerating summary comparison tables...")
        
        # Combine all results into one DataFrame
        all_summary_data = []
        
        if ml_data:
            for n_categories, results in ml_data.items():
                for entry in results:
                    summary_entry = {
                        'Categories': n_categories,
                        'Model_Type': 'ML',
                        'Model': entry['model'],
                        'Feature': entry['feature_type'],
                        'Accuracy': entry['accuracy'],
                        'Precision': entry['precision'],
                        'Recall': entry['recall'],
                        'F1-Score': entry['f1_score'],
                        'Top-1': entry.get('top1_accuracy', entry['accuracy']),
                        'Top-3': entry.get('top3_accuracy', 0),
                        'Top-5': entry.get('top5_accuracy', 0),
                        'Training_Time': entry.get('training_time', 0),
                        'Inference_Time': entry.get('inference_time', 0)
                    }
                    all_summary_data.append(summary_entry)
        
        if dl_data:
            for n_categories, results in dl_data.items():
                for entry in results:
                    summary_entry = {
                        'Categories': n_categories,
                        'Model_Type': 'DL',
                        'Model': entry['model'],
                        'Feature': entry['feature_type'],
                        'Accuracy': entry['accuracy'],
                        'Precision': entry['precision'],
                        'Recall': entry['recall'],
                        'F1-Score': entry['f1_score'],
                        'Top-1': entry.get('top1_accuracy', entry['accuracy']),
                        'Top-3': entry.get('top3_accuracy', 0),
                        'Top-5': entry.get('top5_accuracy', 0),
                        'Training_Time': entry.get('training_time', 0),
                        'Inference_Time': entry.get('inference_time', 0)
                    }
                    all_summary_data.append(summary_entry)
        
        if all_summary_data:
            summary_df = pd.DataFrame(all_summary_data)
            summary_df = summary_df.round(4)
            
            # Save comprehensive summary
            summary_path = self.overall_dir / "Overall_Performance_Summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"Overall summary table saved: {summary_path}")
            
            # Generate best performers summary
            print("\nBest Overall Performers:")
            for metric in ['Accuracy', 'F1-Score', 'Top-1', 'Top-3', 'Top-5']:
                if metric in summary_df.columns:
                    best = summary_df.loc[summary_df[metric].idxmax()]
                    print(f"  {metric}: {best['Model']} ({best['Model_Type']}, {best['Feature']}) "
                          f"on {best['Categories']} categories = {best[metric]:.4f}")
            
            # Best by model type
            print(f"\nBest ML Model:")
            ml_best = summary_df[summary_df['Model_Type'] == 'ML']
            if len(ml_best) > 0:
                ml_best_row = ml_best.loc[ml_best['Top-1'].idxmax()]
                print(f"  {ml_best_row['Model']} ({ml_best_row['Feature']}) on {ml_best_row['Categories']} categories")
                print(f"  Top-1: {ml_best_row['Top-1']:.4f}, F1: {ml_best_row['F1-Score']:.4f}")
            
            print(f"\nBest DL Model:")
            dl_best = summary_df[summary_df['Model_Type'] == 'DL']
            if len(dl_best) > 0:
                dl_best_row = dl_best.loc[dl_best['Top-1'].idxmax()]
                print(f"  {dl_best_row['Model']} ({dl_best_row['Feature']}) on {dl_best_row['Categories']} categories")
                print(f"  Top-1: {dl_best_row['Top-1']:.4f}, F1: {dl_best_row['F1-Score']:.4f}")
        
        return summary_df if all_summary_data else None
    
    def generate_all_comparisons(self):
        """Generate all overall comparison visualizations"""
        print("Starting Overall Performance Analysis...")
        
        # Load all results
        ml_data, dl_data = self.load_all_results()
        
        if not ml_data and not dl_data:
            print("No ML or DL results found. Run training phases first.")
            return
        
        # Combine data for plotting
        combined_metrics = self.combine_results_for_plotting(ml_data, dl_data)
        
        if combined_metrics:
            # Generate all visualizations
            self.generate_combined_line_plots(combined_metrics)
            self.generate_combined_topk_plots(combined_metrics)
            self.generate_combined_bar_plots(ml_data, dl_data)
            self.generate_combined_radar_plots(ml_data, dl_data)
            self.generate_summary_comparison(ml_data, dl_data)
            
            print(f"\nAll overall comparison visualizations saved to: {self.overall_dir}")
        else:
            print("No valid data found for comparison plots.")

def main():
    """Main function to run overall comparison analysis"""
    analyzer = OverallPerformanceAnalyzer()
    analyzer.generate_all_comparisons()

if __name__ == "__main__":
    main()