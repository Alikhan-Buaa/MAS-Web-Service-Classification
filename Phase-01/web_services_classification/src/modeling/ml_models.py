"""
Machine Learning Models for Web Service Classification
Implements Logistic Regression, Random Forest, and XGBoost models
"""

import pandas as pd
import numpy as np
import joblib
import logging
import time
import json
import yaml
import pickle  # Added for saving ml_final_results
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Import configuration
from src.config import (
    ML_CONFIG, DATA_CONFIG, PREPROCESSING_CONFIG,
    CATEGORY_SIZES, RESULTS_CONFIG, SAVED_MODELS_CONFIG
)
from src.preprocessing.feature_extraction import FeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MLModelTrainer:
    """Main class for training machine learning models"""

    def __init__(self):
        self.models = {}
        self.feature_extractor = FeatureExtractor()
        # Added: Initialize storage for final results
        self.ml_final_results = {}

    def create_models(self):
        """Create ML model instances with configurations"""
        return {
            "LogisticRegression": LogisticRegression(**ML_CONFIG["logistic_regression"]),
            "RandomForest": RandomForestClassifier(**ML_CONFIG["random_forest"]),
            "XGBoost": xgb.XGBClassifier(**ML_CONFIG["xgboost"])
        }

    def calculate_top_k_accuracy(self, y_true, y_proba, k=5):
        """Calculate Top-K accuracy"""
        if k == 1:
            y_pred = np.argmax(y_proba, axis=1)
            return accuracy_score(y_true, y_pred)
        top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
        correct = sum(true in top_k for true, top_k in zip(y_true, top_k_preds))
        return correct / len(y_true)

    def generate_confusion_heatmap(self, cm, class_labels, model_name, n_categories, feature_type="tfidf"):
        """Generate confusion matrix heatmap with dynamic sizing"""
        import math
        
        n_classes = len(class_labels)
        
        # Dynamic figure size based on number of classes
        figsize = (12, 10)
        if n_classes >= 40:
            figsize = (18, 18)
        elif n_classes > 20:
            figsize = (14, 12)
        
        # Adjust annotation font size based on number of classes
        annot_fontsize = 8 if n_classes <= 20 else 6 if n_classes <= 40 else 4
        
        fig = plt.figure(figsize=figsize)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        
        # Shorten labels if too long
        display_labels = [lbl[:12] + "..." if len(lbl) > 15 else lbl for lbl in class_labels]
        
        sns.heatmap(
            cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=display_labels, yticklabels=display_labels,
            annot_kws={"size": annot_fontsize}
        )
        
        plt.title(f"Confusion Matrix: {model_name} ({feature_type.upper()}, Top-{n_categories})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        
        # Save to correct directory
        cm_dir = RESULTS_CONFIG["ml_category_paths"][n_categories]
        cm_dir.mkdir(parents=True, exist_ok=True)
        save_path = cm_dir / f"{model_name}_{feature_type}_top_{n_categories}_categories_confusion_matrix.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return save_path

    def generate_classification_report_csv(self, y_true, y_pred, class_labels, model_name, n_categories, feature_type="tfidf"):
        """Generate classification report with accuracy per class"""
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        df = pd.DataFrame(report_dict).transpose()

        cm = confusion_matrix(y_true, y_pred)
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
        df["accuracy"] = np.concatenate([per_class_acc, [None] * (len(df) - len(per_class_acc))])
        df["category_name"] = df.index.map(
            lambda x: class_labels[int(x)] if x.isdigit() and int(x) < len(class_labels) else x
        )

        df.reset_index(inplace=True)
        df.rename(columns={"index": "Class"}, inplace=True)

        reports_dir = RESULTS_CONFIG["ml_category_paths"][n_categories]
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir / f"{model_name}_{feature_type}_top_{n_categories}_categories_classification_report.csv"
        df.to_csv(out_path, index=False)
        return out_path

    # Added: New method to save model performance data for plotting
    def save_model_performance_data(self, results, model_name, n_categories, feature_type,top1=None,top3=None,top5=None):
        """Save model performance data in the format needed for plotting"""
        # Create standardized model entry
        model_entry = {
            "model": f"{model_name}-{feature_type}",
            "accuracy": results["accuracy"],
            "precision": results["macro_precision"],
            "recall": results["macro_recall"],
            "f1_score": results["macro_f1"],
            "feature_type": feature_type,
            "n_categories": n_categories,
            "top1_accuracy": round(float(top1),2),
            "top3_accuracy": round(float(top3),2),
            "top5_accuracy": round(float(top5),2)
        }
        
        # Store in ml_final_results structure
        if n_categories not in self.ml_final_results:
            self.ml_final_results[n_categories] = []
        
        self.ml_final_results[n_categories].append(model_entry)
        
        # Save after each model
        self.save_final_results()
        
        return model_entry

    # Added: New method to save final results to pickle file
    def save_final_results(self):
        """Save final results to pickle file for plotting"""
        results_dir = SAVED_MODELS_CONFIG["ml_models_path"]
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "ml_final_results.pkl"
        with open(results_file, "wb") as f:
            pickle.dump(self.ml_final_results, f)
        logger.info(f"ML final results saved to {results_file}")

        results_dir = RESULTS_CONFIG["ml_comparisons_path"]
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "ml_final_results.pkl"
        with open(results_file, "wb") as f:
            pickle.dump(self.ml_final_results, f)
        logger.info(f"ML final results saved to {results_file}")

    def print_model_metrics(self, results, model_name, n_categories, feature_type, training_time):
        """Print model metrics to console in a formatted way"""
        print(f"\n{'='*60}")
        print(f"MODEL PERFORMANCE SUMMARY")
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

    def evaluate_model(self, model, X_test, y_test, model_name, n_categories, feature_type, class_labels):
        """Evaluate model and generate visualizations"""
        start = time.time()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        inference_time = time.time() - start

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)

        top1, top3, top5 = (
            self.calculate_top_k_accuracy(y_test, y_proba, k=1),
            self.calculate_top_k_accuracy(y_test, y_proba, k=3),
            self.calculate_top_k_accuracy(y_test, y_proba, k=5)
        )

        cm = confusion_matrix(y_test, y_pred)
        cm_path = self.generate_confusion_heatmap(cm, class_labels, model_name, n_categories, feature_type)
        report_path = self.generate_classification_report_csv(y_test, y_pred, class_labels, model_name, n_categories, feature_type)

        results = {
            "model_name": model_name,
            "accuracy": float(acc),
            "top1_accuracy": float(top1),
            "top3_accuracy": float(top3),
            "top5_accuracy": float(top5),
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)),
            "confusion_matrix_path": str(cm_path),
            "classification_report_path": str(report_path),
            "inference_time": float(inference_time)
        }

        return results

    def train_model_on_category(self, n_categories, feature_type="tfidf"):
        """Train and evaluate ML models for given category size"""
        logger.info(f"Training models for top_{n_categories}_categories ({feature_type})")

        splits_dir = Path(PREPROCESSING_CONFIG["splits"].format(n=n_categories))
        train_df = pd.read_csv(splits_dir / "train.csv")
        test_df = pd.read_csv(splits_dir / "test.csv")

        labels_file = Path(PREPROCESSING_CONFIG["labels"].format(n=n_categories))
        try:
            with open(labels_file, "r") as f:
                class_labels = yaml.safe_load(f).get("categories", [])
        except FileNotFoundError:
            class_labels = []

        if feature_type == "tfidf":
            self.feature_extractor.load_tfidf_vectorizer(n_categories)
            X_train = self.feature_extractor.tfidf_vectorizer.transform(train_df["cleaned_text"])
            X_test = self.feature_extractor.tfidf_vectorizer.transform(test_df["cleaned_text"])
        else:
            X_train = self.feature_extractor.load_sbert_features(n_categories, "train")
            X_test = self.feature_extractor.load_sbert_features(n_categories, "test")

        y_train, y_test = train_df["encoded_label"], test_df["encoded_label"]

        results = {}
        for model_name, model in self.create_models().items():
            print(f"\nTraining {model_name} with {feature_type.upper()} features on top_{n_categories}_categories...")
            
            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            res = self.evaluate_model(model, X_test, y_test, model_name, n_categories, feature_type, class_labels)
            res["training_time"] = train_time
            res["n_categories"] = n_categories
            res["feature_type"] = feature_type
            results[model_name] = res

            # Print metrics to console
            self.print_model_metrics(res, model_name, n_categories, feature_type, train_time)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            top1, top3, top5 = (
                self.calculate_top_k_accuracy(y_test, y_proba, k=1),
                self.calculate_top_k_accuracy(y_test, y_proba, k=3),
                self.calculate_top_k_accuracy(y_test, y_proba, k=5)
            )
            # Added: Save performance data for plotting
            self.save_model_performance_data(res, model_name, n_categories, feature_type,top1,top3,top5)

            # Save model
            model_dir = SAVED_MODELS_CONFIG["ml_models_path"] / f"top_{n_categories}_categories"
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_dir / f"{model_name}_{feature_type}_top_{n_categories}_categories.pkl")


        return results

    def train_all_categories(self, feature_types=["tfidf", "sbert"]):
        """Train across all category sizes"""
        all_results = {}
        
        print(f"\n{'='*80}")
        print(f"STARTING ML MODEL TRAINING PIPELINE")
        print(f"{'='*80}")
        print(f"Category sizes: {CATEGORY_SIZES}")
        print(f"Feature types: {feature_types}")
        print(f"Models: {ML_CONFIG['models']}")
        print(f"{'='*80}")
        
        for feature in feature_types:
            print(f"\n{'-'*60}")
            print(f"TRAINING WITH {feature.upper()} FEATURES")
            print(f"{'-'*60}")
            
            all_results[feature] = {}
            for n in CATEGORY_SIZES:
                try:
                    print(f"\n>>> Processing top_{n}_categories with {feature.upper()} features...")
                    all_results[feature][n] = self.train_model_on_category(n, feature)
                    logger.info(f"Successfully completed training for top_{n}_categories with {feature}")
                except Exception as e:
                    logger.error(f"Error training {feature} on top_{n}_categories: {e}")
                    print(f"ERROR: Failed to train {feature} on top_{n}_categories: {e}")
        
        print(f"\n{'='*80}")
        print(f"ML MODEL TRAINING PIPELINE COMPLETED")
        print(f"{'='*80}")
        
        # Added: Print summary of saved data
        print(f"\nML Final Results Summary:")
        print(f"  File location: {SAVED_MODELS_CONFIG['ml_models_path']}/ml_final_results.pkl")
        print(f"  Categories processed: {list(self.ml_final_results.keys())}")
        print(f"  Total model entries: {sum(len(models) for models in self.ml_final_results.values())}")
        
        return all_results

    def plot_and_analyze_results(self, results_file_path=None, charts_dir=None):
        """
        Generate comprehensive plots and analysis for ML models
        
        Args:
            results_file_path (str, optional): Path to the ml_final_results.pkl file. 
                                            If None, uses config path.
            charts_dir (str, optional): Directory to save charts. 
                                    If None, uses config path.
        """
        
        # Use config paths instead of hard-coded ones
        if results_file_path is None:
            results_file_path = RESULTS_CONFIG["ml_comparisons_path"] / "ml_final_results.pkl"
        else:
            results_file_path = Path(results_file_path)
        
        if charts_dir is None:
            charts_dir = RESULTS_CONFIG["ml_comparisons_path"] / "charts"
        else:
            charts_dir = Path(charts_dir)
        
        if not results_file_path.exists():
            print(f"No ML results file found at: {results_file_path}")
            return
        
        # Create charts directory
        charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        with open(results_file_path, "rb") as f:
            ml_final_results = pickle.load(f)
        
        print(f"Generating plots and analysis for ML results...")
        print(f"Results loaded from: {results_file_path}")
        print(f"Charts will be saved to: {charts_dir}")
        
        model_metrics = {}

        # Parse results and organize by model and feature type
        for n, results in ml_final_results.items():
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
                        # Added top-K accuracy metrics
                        'top1_accuracy': [],
                        'top3_accuracy': [],
                        'top5_accuracy': []
                    }
                
                model_metrics[model][feature_type]['n'].append(entry['n_categories'])
                model_metrics[model][feature_type]['accuracy'].append(entry['accuracy'])
                model_metrics[model][feature_type]['precision'].append(entry['precision'])
                model_metrics[model][feature_type]['recall'].append(entry['recall'])
                model_metrics[model][feature_type]['f1_score'].append(entry['f1_score'])
                
                # Add top-K accuracy metrics
                model_metrics[model][feature_type]['top1_accuracy'].append(entry.get('top1_accuracy', entry['accuracy']))
                model_metrics[model][feature_type]['top3_accuracy'].append(entry.get('top3_accuracy', 0))
                model_metrics[model][feature_type]['top5_accuracy'].append(entry.get('top5_accuracy', 0))
            
        # =================================================================
        # LINE PLOTS - Performance vs Category Size
        # =================================================================
        def plot_metric(metric_name, ylabel=None):
            plt.figure(figsize=(12, 6))
            
            for model, features in model_metrics.items():
                for feature_type, data in features.items():
                    label = f"{model} ({feature_type.upper()})"
                    plt.plot(data['n'], data[metric_name], marker='o', label=label, linewidth=2)
            
            plt.title(f'{ylabel or metric_name.replace("_", " ").title()} vs Number of Web Service Categories')
            plt.xlabel('Number of Web Service Categories')
            plt.ylabel(ylabel or metric_name.replace("_", " ").title())
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plot_path = charts_dir / f"ML_Model_Performance_{metric_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Line plot saved: {plot_path}")
            plt.close()

        # Generate line plots for all metrics including top-K
        print("\nGenerating line plots...")
        metrics_config = {
            'accuracy': 'Accuracy',
            'precision': 'Precision (Macro)',
            'recall': 'Recall (Macro)',
            'f1_score': 'F1-Score (Macro)',
            'top1_accuracy': 'Top-1 Accuracy',
            'top3_accuracy': 'Top-3 Accuracy', 
            'top5_accuracy': 'Top-5 Accuracy'
        }
        
        for metric, ylabel in metrics_config.items():
            plot_metric(metric, ylabel)
        
        # =================================================================
        # COMBINED TOP-K ACCURACY PLOT
        # =================================================================
        print("\nGenerating combined top-K accuracy plot...")
        plt.figure(figsize=(14, 8))
        
        for model, features in model_metrics.items():
            for feature_type, data in features.items():
                label_base = f"{model} ({feature_type.upper()})"
                
                plt.plot(data['n'], data['top1_accuracy'], marker='o', label=f"{label_base} - Top-1", linewidth=2)
                plt.plot(data['n'], data['top3_accuracy'], marker='s', label=f"{label_base} - Top-3", linewidth=2, linestyle='--')
                plt.plot(data['n'], data['top5_accuracy'], marker='^', label=f"{label_base} - Top-5", linewidth=2, linestyle=':')
        
        plt.title('Top-K Accuracy Comparison Across All Models')
        plt.xlabel('Number of Web Service Categories')
        plt.ylabel('Top-K Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plot_path = charts_dir / "ML_Model_Performance_topk_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Combined top-K plot saved: {plot_path}")
        plt.close()
        
        # =================================================================
        # BAR PLOTS - Performance by Category Size (Enhanced)
        # =================================================================
        print("\nGenerating enhanced bar plots for each category size...")
        
        for n in CATEGORY_SIZES:
            if n not in ml_final_results:
                print(f"⚠️ Skipping n={n} (no results found)")
                continue

            combined_results = ml_final_results[n]
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
            ax1.set_title(f'Standard Performance Metrics - Top {n} Categories', fontsize=14)
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
                ax2.set_title(f'Top-K Accuracy Metrics - Top {n} Categories', fontsize=14)
                ax2.set_ylabel('Top-K Accuracy')
                ax2.set_ylim(0.3, 1.0)
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(axis='y', alpha=0.3)
                ax2.legend(title='Top-K Metric')
            
            plt.tight_layout()
            
            # Save enhanced plot
            plot_path = charts_dir / f"ML_Model_Performance_enhanced_top_{n}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Enhanced bar plot saved: {plot_path}")
            plt.close()

        # =================================================================
        # SUMMARY STATISTICS TABLE
        # =================================================================
        print("\nGenerating summary statistics...")
        
        # Create summary table
        summary_data = []
        for n in CATEGORY_SIZES:
            if n in ml_final_results:
                for entry in ml_final_results[n]:
                    summary_data.append({
                        'Categories': n,
                        'Model': entry['model'],
                        'Feature': entry['feature_type'],
                        'Accuracy': entry['accuracy'],
                        'F1-Score': entry['f1_score'],
                        'Top-1': entry.get('top1_accuracy', entry['accuracy']),
                        'Top-3': entry.get('top3_accuracy', 0),
                        'Top-5': entry.get('top5_accuracy', 0)
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.round(4)
        
        # Save summary table
        summary_path = charts_dir / "ML_Model_Performance_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary table saved: {summary_path}")
        
        # Display best performing models
        print("\nTop performing models by metric:")
        for metric in ['Accuracy', 'F1-Score', 'Top-1', 'Top-3', 'Top-5']:
            if metric in summary_df.columns:
                best = summary_df.loc[summary_df[metric].idxmax()]
                print(f"  {metric}: {best['Model']} ({best['Feature']}) on {best['Categories']} categories = {best[metric]:.4f}")

    def plot_ml_results_only(self):
        """Convenience function to plot ML results with config paths"""
        self.plot_and_analyze_results()  # Now uses config paths by default


def main():
    trainer = MLModelTrainer()
    results = trainer.train_all_categories()
    out_file = SAVED_MODELS_CONFIG["ml_models_path"] / "ml_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()