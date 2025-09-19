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
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import xgboost as xgb

# Import configuration
from src.config import (
    ML_CONFIG, PREPROCESSING_CONFIG,
    CATEGORY_SIZES, SAVED_MODELS_CONFIG
)
from src.preprocessing.feature_extraction import FeatureExtractor
from src.evaluation.evaluate import ModelEvaluator  # Import common evaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MLModelTrainer:
    """Main class for training machine learning models"""

    def __init__(self):
        self.models = {}
        self.feature_extractor = FeatureExtractor()
        self.evaluator = ModelEvaluator()  # Use common evaluator

    def create_models(self):
        """Create ML model instances with configurations"""
        return {
            "LogisticRegression": LogisticRegression(**ML_CONFIG["logistic_regression"]),
            "RandomForest": RandomForestClassifier(**ML_CONFIG["random_forest"]),
            "XGBoost": xgb.XGBClassifier(**ML_CONFIG["xgboost"])
        }

    def evaluate_model(self, model, X_test, y_test, model_name, n_categories, feature_type, class_labels):
        """Evaluate model and generate visualizations"""
        start = time.time()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        inference_time = time.time() - start

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)

        # Calculate top-K accuracies using common evaluator
        top1 = self.evaluator.calculate_top_k_accuracy(y_test, y_proba, k=1)
        top3 = self.evaluator.calculate_top_k_accuracy(y_test, y_proba, k=3)
        top5 = self.evaluator.calculate_top_k_accuracy(y_test, y_proba, k=5)

        cm = confusion_matrix(y_test, y_pred)
        
        # Use common evaluator methods
        cm_path = self.evaluator.generate_confusion_heatmap(cm, class_labels, model_name, n_categories, feature_type, "ml")
        report_path = self.evaluator.generate_classification_report_csv(y_test, y_pred, class_labels, model_name, n_categories, feature_type, "ml")

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

        # Load class labels using common evaluator
        class_labels = self.evaluator.load_class_labels(n_categories)

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

            # Print metrics using common evaluator
            self.evaluator.print_model_metrics(res, model_name, n_categories, feature_type, train_time, "ML")

            # Save performance data using common evaluator
            self.evaluator.save_model_performance_data(res, model_name, n_categories, feature_type, "ml")

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
        
        # Print summary of saved data
        print(f"\nML Final Results Summary:")
        print(f"  File location: {SAVED_MODELS_CONFIG['ml_models_path']}/ml_final_results.pkl")
        print(f"  Categories processed: {list(self.evaluator.final_results.keys())}")
        print(f"  Total model entries: {sum(len(models) for models in self.evaluator.final_results.values())}")
        
        # Automatically generate all visualizations
        print(f"\n{'='*80}")
        print(f"GENERATING ML VISUALIZATIONS")
        print(f"{'='*80}")
        try:
            print("Generating line plots, bar plots, and summary statistics...")
            self.plot_ml_results_only()
            print("Generating radar plots...")
            self.generate_ml_radar_plots_only()
            print("All ML visualizations completed successfully!")
        except Exception as e:
            logger.error(f"Error generating ML visualizations: {e}")
            print(f"Warning: Some visualizations may not have been generated due to errors.")
        
        return all_results

    def plot_ml_results_only(self):
        """Convenience function to plot ML results with config paths"""
        from src.config import RESULTS_CONFIG
        
        results_file_path = RESULTS_CONFIG["ml_comparisons_path"] / "ml_final_results.pkl"
        charts_dir = RESULTS_CONFIG["ml_comparisons_path"] / "charts"
        
        self.evaluator.plot_results_comparison(results_file_path, charts_dir, "ml")
    
    def plot_ml_results_with_radar(self, show_plots=False):
        """Generate comprehensive ML plots including radar charts"""
        from src.config import RESULTS_CONFIG
        
        results_file_path = RESULTS_CONFIG["ml_comparisons_path"] / "ml_final_results.pkl"
        charts_dir = RESULTS_CONFIG["ml_comparisons_path"] / "charts"
        
        self.evaluator.plot_results_with_radar(results_file_path, charts_dir, "ml", show_plots)
    
    def generate_ml_radar_plots_only(self, show_plots=False):
        """Generate only radar plots for ML models"""
        self.evaluator.generate_radar_plots("ml", show_plots)


def main():
    trainer = MLModelTrainer()
    results = trainer.train_all_categories()
    
    # Save final results
    out_file = SAVED_MODELS_CONFIG["ml_models_path"] / "ml_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()