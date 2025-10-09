"""
Enhanced Overall Performance Comparison Module
Combines ML, DL, BERT, DeepSeek, and Fusion model results for comprehensive analysis
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
    """Enhanced analyzer for combined ML, DL, BERT, DeepSeek, and Fusion model performance"""
    
    def __init__(self):
        # Create overall results directory
        self.overall_dir = RESULTS_CONFIG.get("overall_results_path", Path("results") / "overall")
        self.overall_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced model naming patterns
        self.naming_patterns = {
            "logistic_regression": "LogisticRegression",
            "random_forest": "RandomForest", 
            "xgboost": "XGBoost",
            "bilstm": "BiLSTM",
            "roberta_base": "RoBERTa-Base",
            "roberta_large": "RoBERTa-Large",
            "deepseek_7b_base": "DeepSeek-7B-Base",
            # Fusion patterns
            "deepseek_roberta_fusion": "DeepSeek-RoBERTa-Fusion",
            "deepseek_roberta_fusion_concat": "DeepSeek-RoBERTa-Fusion-Concat",
            "deepseek_roberta_fusion_average": "DeepSeek-RoBERTa-Fusion-Average",
            "deepseek_roberta_fusion_weighted": "DeepSeek-RoBERTa-Fusion-Weighted",
            "deepseek_roberta_fusion_gating": "DeepSeek-RoBERTa-Fusion-Gating"
        }
        
    def load_all_results(self):
        """Load ML, DL, BERT, DeepSeek, and Fusion results"""
        results = {
            'ml': None,
            'dl': None, 
            'bert': None,
            'deepseek': None,
            'fusion': None
        }
        
        # Define result files for each model type
        result_files = {
            'ml': RESULTS_CONFIG["ml_comparisons_path"] / "ml_final_results.pkl",
            'dl': RESULTS_CONFIG["dl_comparisons_path"] / "dl_final_results.pkl",
            'bert': RESULTS_CONFIG["bert_comparisons_path"] / "bert_final_results.pkl",
            'deepseek': RESULTS_CONFIG["deepseek_comparisons_path"] / "deepseek_final_results.pkl",
            'fusion': RESULTS_CONFIG["fusion_comparisons_path"] / "fusion_final_results.pkl"
        }
        
        # Load each result type
        for model_type, file_path in result_files.items():
            try:
                if file_path.exists():
                    with open(file_path, "rb") as f:
                        data = pickle.load(f)
                    results[model_type] = data
                    logger.info(f"Loaded {model_type.upper()} results: {len(data)} category sizes")
                else:
                    logger.warning(f"{model_type.upper()} results file not found: {file_path}")
            except Exception as e:
                logger.error(f"Error loading {model_type.upper()} results: {e}")
        
        return results
    
    def normalize_data_structure(self, data, model_type):
        """Normalize different data structures to a common format"""
        normalized = {}
        
        if not data:
            return normalized
        
        try:
            for n_categories, category_data in data.items():
                normalized[n_categories] = []
                
                if model_type in ['ml', 'dl']:
                    # ML/DL format: list of dictionaries per category
                    if isinstance(category_data, list):
                        for entry in category_data:
                            normalized_entry = self._normalize_entry(entry, model_type)
                            if normalized_entry:
                                normalized[n_categories].append(normalized_entry)
                    elif isinstance(category_data, dict):
                        # Handle nested dictionary format
                        for model_key, model_data in category_data.items():
                            normalized_entry = self._normalize_entry(model_data, model_type, model_key)
                            if normalized_entry:
                                normalized[n_categories].append(normalized_entry)
                
                elif model_type in ['bert', 'deepseek', 'fusion']:
                    # BERT/DeepSeek/Fusion format: dictionary with model_feature keys
                    if isinstance(category_data, dict):
                        for model_key, model_data in category_data.items():
                            normalized_entry = self._normalize_entry(model_data, model_type, model_key)
                            if normalized_entry:
                                normalized[n_categories].append(normalized_entry)
        
        except Exception as e:
            logger.error(f"Error normalizing {model_type} data: {e}")
        
        return normalized
    
    def _normalize_entry(self, entry, model_type, model_key=None):
        """Normalize a single entry to common format"""
        try:
            # Extract model name and feature type
            if model_key:
                # For BERT/DeepSeek/Fusion: model_key like "RoBERTa_Base_raw_text" or "DeepSeek_RoBERTa_Fusion_Concat"
                parts = model_key.split('_')
                if len(parts) >= 2:
                    model_name = '_'.join(parts[:-1])  # Everything except last part
                    feature_type = parts[-1]  # Last part is feature type
                else:
                    model_name = model_key
                    feature_type = entry.get('feature_type', 'raw_text')
            else:
                # For ML/DL: extract from entry
                model_name = entry.get('model', entry.get('model_name', 'Unknown'))
                feature_type = entry.get('feature_type', 'unknown')
            
            # Create normalized entry
            normalized = {
                'model': model_name,
                'model_type': model_type.upper(),
                'feature_type': feature_type,
                'n_categories': entry.get('n_categories', 0),
                'accuracy': entry.get('accuracy', entry.get('top1_accuracy', 0)),
                'precision': entry.get('precision', entry.get('macro_precision', 0)),
                'recall': entry.get('recall', entry.get('macro_recall', 0)),
                'f1_score': entry.get('f1_score', entry.get('macro_f1', 0)),
                'top1_accuracy': entry.get('top1_accuracy', entry.get('accuracy', 0)),
                'top3_accuracy': entry.get('top3_accuracy', 0),
                'top5_accuracy': entry.get('top5_accuracy', 0),
                'training_time': entry.get('training_time', 0),
                'inference_time': entry.get('inference_time', 0)
            }
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing entry: {e}")
            return None
    
    def combine_results_for_plotting(self, all_results):
        """Combine all model results into unified structure for plotting"""
        combined_metrics = {}
        
        # Process each model type
        for model_type, data in all_results.items():
            if not data:
                continue
                
            # Normalize data structure
            normalized_data = self.normalize_data_structure(data, model_type)
            
            # Process normalized data
            for n_categories, results in normalized_data.items():
                for entry in results:
                    model_key = f"{entry['model']} ({entry['model_type']})"
                    feature_type = entry['feature_type']
                    
                    if model_key not in combined_metrics:
                        combined_metrics[model_key] = {}
                    if feature_type not in combined_metrics[model_key]:
                        combined_metrics[model_key][feature_type] = {
                            'n': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
                            'top1_accuracy': [], 'top3_accuracy': [], 'top5_accuracy': [],
                            'training_time': [], 'inference_time': [], 'model_type': entry['model_type']
                        }
                    
                    # Append data
                    metrics = combined_metrics[model_key][feature_type]
                    metrics['n'].append(entry['n_categories'])
                    metrics['accuracy'].append(entry['accuracy'])
                    metrics['precision'].append(entry['precision'])
                    metrics['recall'].append(entry['recall'])
                    metrics['f1_score'].append(entry['f1_score'])
                    metrics['top1_accuracy'].append(entry['top1_accuracy'])
                    metrics['top3_accuracy'].append(entry['top3_accuracy'])
                    metrics['top5_accuracy'].append(entry['top5_accuracy'])
                    metrics['training_time'].append(entry['training_time'])
                    metrics['inference_time'].append(entry['inference_time'])
        
        return combined_metrics
    
    def generate_combined_line_plots(self, combined_metrics):
        """Generate line plots comparing all model types"""
        print("\nGenerating combined line plots for all model types...")
        
        # Define colors for different model types
        model_colors = {
            'ML': ['#1f77b4', '#ff7f0e', '#2ca02c'],      # Blue tones
            'DL': ['#d62728', '#9467bd', '#8c564b'],      # Red/purple tones  
            'BERT': ['#e377c2', '#7f7f7f', '#bcbd22'],    # Pink/gray tones
            'DEEPSEEK': ['#17becf', '#ff9896', '#c5b0d5'], # Cyan/light tones
            'FUSION': ['#28a745', '#ffc107', '#dc3545', '#6c757d']  # Green/yellow/red/gray for 4 fusion types
        }
        
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
            plt.figure(figsize=(16, 10))
            
            color_indices = {'ML': 0, 'DL': 0, 'BERT': 0, 'DEEPSEEK': 0, 'FUSION': 0}
            
            for model, features in combined_metrics.items():
                for feature_type, data in features.items():
                    if len(data['n']) == 0:
                        continue
                        
                    label = f"{model} ({feature_type.upper()})"
                    model_type = data['model_type']
                    
                    # Choose color and style based on model type
                    colors = model_colors.get(model_type, ['#000000'])
                    color = colors[color_indices[model_type] % len(colors)]
                    color_indices[model_type] += 1
                    
                    # Different line styles for different model types
                    linestyles = {'ML': '-', 'DL': '--', 'BERT': '-.', 'DEEPSEEK': ':', 'FUSION': '-'}
                    linestyle = linestyles.get(model_type, '-')
                    
                    plt.plot(data['n'], data[metric], marker='o', label=label, 
                            linewidth=2.5, color=color, linestyle=linestyle, markersize=6)
            
            plt.title(f'Overall Model Comparison: {ylabel} vs Number of Categories', fontsize=16, fontweight='bold')
            plt.xlabel('Number of Web Service Categories', fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.tight_layout()
            
            plot_path = self.overall_dir / f"Overall_Comparison_{metric}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Combined line plot saved: {plot_path}")
            plt.close()
    
    def generate_summary_comparison(self, all_results):
        """Generate enhanced summary comparison tables"""
        print("\nGenerating comprehensive summary comparison tables...")
        
        # Combine all results into one DataFrame
        all_summary_data = []
        
        for model_type, data in all_results.items():
            if not data:
                continue
                
            normalized_data = self.normalize_data_structure(data, model_type)
            
            for n_categories, results in normalized_data.items():
                for entry in results:
                    summary_entry = {
                        'Categories': n_categories,
                        'Model_Type': entry['model_type'],
                        'Model': entry['model'],
                        'Feature': entry['feature_type'],
                        'Accuracy': entry['accuracy'],
                        'Precision': entry['precision'],
                        'Recall': entry['recall'],
                        'F1-Score': entry['f1_score'],
                        'Top-1': entry['top1_accuracy'],
                        'Top-3': entry['top3_accuracy'],
                        'Top-5': entry['top5_accuracy'],
                        'Training_Time': entry['training_time'],
                        'Inference_Time': entry['inference_time']
                    }
                    all_summary_data.append(summary_entry)
        
        if all_summary_data:
            summary_df = pd.DataFrame(all_summary_data)
            summary_df = summary_df.round(4)
            
            # Save comprehensive summary
            summary_path = self.overall_dir / "Overall_Performance_Summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"✓ Overall summary table saved: {summary_path}")
            
            # Generate enhanced analysis
            print(f"\n{'='*80}")
            print("COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
            print(f"{'='*80}")
            
            # Best performers by metric
            print("\nBest Overall Performers by Metric:")
            for metric in ['Accuracy', 'F1-Score', 'Top-1', 'Top-3', 'Top-5']:
                if metric in summary_df.columns and summary_df[metric].max() > 0:
                    best = summary_df.loc[summary_df[metric].idxmax()]
                    print(f"  {metric:12}: {best['Model']:25} ({best['Model_Type']:8}, {best['Feature']:8}) "
                          f"on {best['Categories']:2} categories = {best[metric]:.4f}")
            
            # Best by model type
            print(f"\nBest Performer by Model Type:")
            for model_type in ['ML', 'DL', 'BERT', 'DEEPSEEK', 'FUSION']:
                type_data = summary_df[summary_df['Model_Type'] == model_type]
                if len(type_data) > 0:
                    best_row = type_data.loc[type_data['Top-1'].idxmax()]
                    print(f"  {model_type:8}: {best_row['Model']:25} ({best_row['Feature']:8}) "
                          f"on {best_row['Categories']:2} categories")
                    print(f"           Top-1: {best_row['Top-1']:.4f}, F1: {best_row['F1-Score']:.4f}, "
                          f"Training: {best_row['Training_Time']:.2f}s")
            
            # Feature type analysis
            print(f"\nFeature Type Effectiveness:")
            feature_analysis = summary_df.groupby('Feature').agg({
                'Top-1': 'mean',
                'F1-Score': 'mean',
                'Training_Time': 'mean'
            }).round(4)
            
            for feature in feature_analysis.index:
                row = feature_analysis.loc[feature]
                print(f"  {feature:10}: Avg Top-1: {row['Top-1']:.4f}, "
                      f"Avg F1: {row['F1-Score']:.4f}, Avg Training: {row['Training_Time']:.2f}s")
            
            # Model count summary
            print(f"\nModel Coverage Summary:")
            coverage = summary_df.groupby(['Model_Type', 'Categories']).size().unstack(fill_value=0)
            print(coverage)
            
            print(f"{'='*80}")
            
        else:
            print("No valid data found for summary generation.")
            summary_df = None
        
        return summary_df
    
    def generate_all_comparisons(self):
        """Generate all overall comparison visualizations including BERT, DeepSeek, and Fusion"""
        print("Starting Comprehensive Performance Analysis (ML + DL + BERT + DeepSeek + Fusion)...")
        
        # Load all results
        all_results = self.load_all_results()
        
        # Check if any data was loaded
        has_data = any(data is not None for data in all_results.values())
        if not has_data:
            print("No results found. Run training phases first.")
            return
        
        # Report what was loaded
        loaded_types = [model_type for model_type, data in all_results.items() if data is not None]
        print(f"Loaded results for: {', '.join(loaded_types).upper()}")
        
        # Combine data for plotting
        combined_metrics = self.combine_results_for_plotting(all_results)
        
        if combined_metrics:
            print(f"Combined metrics for {len(combined_metrics)} model configurations")
            
            # Generate all visualizations
            try:
                self.generate_combined_line_plots(combined_metrics)
                print("✓ Line plots generated")
            except Exception as e:
                logger.error(f"Error generating line plots: {e}")
            
            try:
                self.generate_summary_comparison(all_results)
                print("✓ Summary comparison generated")
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
            
            print(f"\nAll overall comparison visualizations saved to: {self.overall_dir}")
        else:
            print("No valid data found for comparison plots.")
            
        # Debug information
        print(f"\nDEBUG: Result file check:")
        for model_type in ['ml', 'dl', 'bert', 'deepseek', 'fusion']:
            file_path = RESULTS_CONFIG[f"{model_type}_comparisons_path"] / f"{model_type}_final_results.pkl"
            status = "EXISTS" if file_path.exists() else "MISSING"
            print(f"  {model_type.upper():8}: {file_path} - {status}")


def main():
    """Main function to run enhanced overall comparison analysis"""
    analyzer = OverallPerformanceAnalyzer()
    analyzer.generate_all_comparisons()


if __name__ == "__main__":
    main()