"""
Fixed ML Explainability POC for 20 News Categories
Handles edge cases and missing data gracefully
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Explainability Libraries
import lime
import lime.lime_tabular
from lime.lime_text import LimeTextExplainer
import shap

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsExplainabilityPOC:
    """Fixed version with proper error handling for 20 categories"""
    
    def __init__(self, n_categories=20, max_features=2000):
        self.n_categories = n_categories
        self.max_features = max_features
        self.models = {}
        self.explainers = {}
        self.results = {
            'shap': {},
            'lime': {},
            'global_insights': {},
            'per_class_insights': {}
        }
        self.setup_directories()
        
    def setup_directories(self):
        """Create directory structure for outputs"""
        self.base_dir = Path("xai_artifacts")
        self.shap_dir = self.base_dir / "shap"
        self.lime_dir = self.base_dir / "lime"
        self.viz_dir = self.base_dir / "visualizations"
        self.reports_dir = self.base_dir / "reports"
        
        for dir_path in [self.shap_dir, self.lime_dir, self.viz_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Created output directories under {self.base_dir}")
    
    def load_and_prepare_data(self):
        """Load news dataset and prepare for training"""
        print("\n" + "="*60)
        print("LOADING 20 NEWSGROUPS DATASET")
        print("="*60)
        
        # Get all 20 newsgroup categories
        all_categories = [
            'alt.atheism',
            'comp.graphics',
            'comp.os.ms-windows.misc',
            'comp.sys.ibm.pc.hardware',
            'comp.sys.mac.hardware',
            'comp.windows.x',
            'misc.forsale',
            'rec.autos',
            'rec.motorcycles',
            'rec.sport.baseball',
            'rec.sport.hockey',
            'sci.crypt',
            'sci.electronics',
            'sci.med',
            'sci.space',
            'soc.religion.christian',
            'talk.politics.guns',
            'talk.politics.mideast',
            'talk.politics.misc',
            'talk.religion.misc'
        ]
        
        # Use subset if specified, otherwise use all 20
        categories = all_categories[:self.n_categories] if self.n_categories < 20 else all_categories
        
        print(f"\nLoading {len(categories)} categories...")
        
        # Load data
        self.newsgroups_train = fetch_20newsgroups(
            subset='train',
            categories=categories,
            remove=('headers', 'footers', 'quotes'),
            random_state=42
        )
        
        self.newsgroups_test = fetch_20newsgroups(
            subset='test',
            categories=categories,
            remove=('headers', 'footers', 'quotes'),
            random_state=42
        )
        
        self.categories = categories
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            max_df=0.9,
            min_df=2
        )
        
        self.X_train = self.vectorizer.fit_transform(self.newsgroups_train.data)
        self.X_test = self.vectorizer.transform(self.newsgroups_test.data)
        self.y_train = self.newsgroups_train.target
        self.y_test = self.newsgroups_test.target
        
        # Get feature names
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"✓ Loaded {len(categories)} categories")
        print(f"✓ Training samples: {self.X_train.shape[0]}")
        print(f"✓ Test samples: {self.X_test.shape[0]}")
        print(f"✓ Features (words): {len(self.feature_names)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple ML models"""
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        models_config = {
            'LogisticRegression': LogisticRegression(
                max_iter=2000,
                random_state=42,
                multi_class='ovr',
                solver='lbfgs'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,  # Reduced for faster training
                max_depth=15,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss',
                objective='multi:softprob',
                num_class=self.n_categories,
                n_jobs=-1
            )
        }
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            try:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                self.models[name] = model
                print(f"✓ {name} - Accuracy: {accuracy:.3f}")
            except Exception as e:
                print(f"✗ Error training {name}: {e}")
        
        return self.models
    
    def setup_explainers(self):
        """Setup LIME and SHAP explainers"""
        print("\n" + "="*60)
        print("SETTING UP EXPLAINERS")
        print("="*60)
        
        for model_name, model in self.models.items():
            print(f"\nSetting up explainers for {model_name}...")
            
            try:
                # LIME Text Explainer
                lime_text = LimeTextExplainer(
                    class_names=self.categories,
                    verbose=False
                )
                
                # LIME Tabular Explainer
                lime_tabular = lime.lime_tabular.LimeTabularExplainer(
                    self.X_train.toarray()[:100],  # Use subset for speed
                    feature_names=self.feature_names,
                    class_names=self.categories,
                    mode='classification',
                    verbose=False
                )
                
                # SHAP Explainer
                if model_name == 'LogisticRegression':
                    shap_explainer = shap.LinearExplainer(model, self.X_train.toarray()[:50])
                elif model_name in ['RandomForest', 'XGBoost']:
                    shap_explainer = shap.TreeExplainer(model)
                else:
                    shap_explainer = shap.KernelExplainer(model.predict_proba, self.X_train.toarray()[:50])
                
                self.explainers[model_name] = {
                    'lime_text': lime_text,
                    'lime_tabular': lime_tabular,
                    'shap': shap_explainer
                }
                
                print(f"✓ {model_name} explainers ready")
            except Exception as e:
                print(f"✗ Error setting up explainers for {model_name}: {e}")
    
    def generate_token_attributions(self, model_name='LogisticRegression', n_examples=5, top_k=10):
        """Generate SHAP and LIME token attributions"""
        print("\n" + "="*60)
        print("GENERATING TOKEN ATTRIBUTIONS")
        print("="*60)
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return [], []
        
        model = self.models[model_name]
        explainer_set = self.explainers[model_name]
        
        lime_results = []
        shap_results = []
        
        for idx in range(min(n_examples, len(self.newsgroups_test.data))):
            print(f"\n--- Example {idx + 1}/{n_examples} ---")
            
            try:
                # Get test instance
                text = self.newsgroups_test.data[idx]
                true_label = self.categories[self.y_test[idx]]
                instance_vectorized = self.X_test[idx]
                
                # Get prediction
                pred_proba = model.predict_proba(instance_vectorized)
                pred_label = self.categories[np.argmax(pred_proba)]
                confidence = np.max(pred_proba)
                
                print(f"True: {true_label} | Predicted: {pred_label} ({confidence:.2%})")
                
                # LIME Explanation
                def predict_fn(texts):
                    return model.predict_proba(self.vectorizer.transform(texts))
                
                lime_exp = explainer_set['lime_text'].explain_instance(
                    text, predict_fn, num_features=top_k, num_samples=500
                )
                
                lime_features = lime_exp.as_list()
                lime_dict = {
                    'example_id': idx,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': float(confidence),
                    'top_tokens': [{'token': t, 'weight': float(w)} for t, w in lime_features[:top_k]]
                }
                lime_results.append(lime_dict)
                
                # SHAP Explanation
                shap_explainer = explainer_set['shap']
                shap_values = shap_explainer.shap_values(instance_vectorized.toarray())
                
                if isinstance(shap_values, list):
                    pred_class_idx = np.argmax(pred_proba)
                    if pred_class_idx < len(shap_values):
                        shap_vals = shap_values[pred_class_idx][0]
                    else:
                        shap_vals = shap_values[0][0]
                else:
                    shap_vals = shap_values[0]
                
                top_indices = np.argsort(np.abs(shap_vals))[-top_k:][::-1]
                
                shap_dict = {
                    'example_id': idx,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': float(confidence),
                    'top_features': [
                        {'feature': self.feature_names[i], 'shap_value': float(shap_vals[i])}
                        for i in top_indices if i < len(self.feature_names)
                    ]
                }
                shap_results.append(shap_dict)
                
            except Exception as e:
                print(f"  ✗ Error processing example {idx}: {e}")
                continue
        
        # Save results
        self.results['lime'][model_name] = lime_results
        self.results['shap'][model_name] = shap_results
        
        # Save to JSON
        try:
            lime_json = self.lime_dir / f"lime_{model_name}_attributions.json"
            shap_json = self.shap_dir / f"shap_{model_name}_attributions.json"
            
            with open(lime_json, 'w') as f:
                json.dump(lime_results, f, indent=2)
            
            with open(shap_json, 'w') as f:
                json.dump(shap_results, f, indent=2)
            
            print(f"\n✓ Saved LIME attributions to {lime_json}")
            print(f"✓ Saved SHAP attributions to {shap_json}")
        except Exception as e:
            print(f"✗ Error saving results: {e}")
        
        return lime_results, shap_results
    
    def aggregate_global_insights(self, model_name='LogisticRegression', n_samples=50):
        """Aggregate global insights with proper error handling"""
        print("\n" + "="*60)
        print("AGGREGATING GLOBAL INSIGHTS")
        print("="*60)
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return [], {}
        
        model = self.models[model_name]
        explainer_set = self.explainers[model_name]
        
        # Sample data
        sample_indices = np.random.choice(
            self.X_test.shape[0], 
            min(n_samples, self.X_test.shape[0]), 
            replace=False
        )
        
        print("\n📊 Computing Global Feature Importance...")
        
        # Collect SHAP values
        shap_explainer = explainer_set['shap']
        all_shap_values = []
        
        for idx in sample_indices[:20]:  # Limit for speed
            try:
                instance = self.X_test[idx]
                shap_vals = shap_explainer.shap_values(instance.toarray())
                
                if isinstance(shap_vals, list):
                    avg_shap = np.mean([sv[0] for sv in shap_vals], axis=0)
                else:
                    avg_shap = shap_vals[0]
                
                all_shap_values.append(np.abs(avg_shap))
            except Exception as e:
                print(f"  Warning: Error processing instance {idx}: {e}")
                continue
        
        if not all_shap_values:
            print("✗ No SHAP values computed")
            return [], {}
        
        # Average absolute SHAP values
        global_importance = np.mean(all_shap_values, axis=0)
        
        # Get top features
        top_global_indices = np.argsort(global_importance)[-25:][::-1]
        top_global_features = [
            (self.feature_names[i], global_importance[i]) 
            for i in top_global_indices if i < len(self.feature_names)
        ]
        
        print("\n🌐 TOP 10 MOST INFLUENTIAL FEATURES:")
        for i, (feature, importance) in enumerate(top_global_features[:10], 1):
            print(f"{i:2d}. '{feature}': {importance:.4f}")
        
        # Per-class importance
        print("\n📈 Computing Per-Class Feature Importance...")
        per_class_importance = {}
        
        for class_idx, class_name in enumerate(self.categories[:10]):  # Limit to first 10 for speed
            class_samples = [i for i in sample_indices if self.y_test[i] == class_idx][:10]
            
            if not class_samples:
                per_class_importance[class_name] = []
                continue
            
            class_shap_values = []
            for idx in class_samples:
                try:
                    instance = self.X_test[idx]
                    shap_vals = shap_explainer.shap_values(instance.toarray())
                    
                    if isinstance(shap_vals, list) and class_idx < len(shap_vals):
                        class_shap = shap_vals[class_idx][0]
                    elif isinstance(shap_vals, list):
                        class_shap = shap_vals[0][0]
                    else:
                        class_shap = shap_vals[0]
                    
                    class_shap_values.append(np.abs(class_shap))
                except:
                    continue
            
            if class_shap_values:
                class_importance = np.mean(class_shap_values, axis=0)
                top_class_indices = np.argsort(class_importance)[-10:][::-1]
                top_class_features = [
                    (self.feature_names[i], class_importance[i]) 
                    for i in top_class_indices if i < len(self.feature_names)
                ]
                per_class_importance[class_name] = top_class_features
            else:
                per_class_importance[class_name] = []
        
        # Save insights
        self.results['global_insights'][model_name] = {
            'overall_top_features': top_global_features,
            'per_class_features': per_class_importance
        }
        
        # Create simple visualization
        self._plot_simple_insights(top_global_features, model_name)
        
        return top_global_features, per_class_importance
    
    def _plot_simple_insights(self, global_features, model_name):
        """Create a simple visualization of global insights"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot top features
        features, importances = zip(*global_features[:20])
        y_pos = np.arange(len(features))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        
        ax.barh(y_pos, importances, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Average Absolute SHAP Value')
        ax.set_title(f'Top 20 Most Influential Features - {model_name}', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        viz_file = self.viz_dir / f"global_insights_{model_name}_simple.png"
        plt.savefig(viz_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved visualization to {viz_file}")
    
    def run_quick_demo(self):
        """Run a quick demonstration"""
        print("\n" + "="*70)
        print(" "*15 + "20-CATEGORY EXPLAINABILITY DEMO")
        print("="*70)
        
        # Load data
        self.load_and_prepare_data()
        
        # Train models
        self.train_models()
        
        # Setup explainers
        self.setup_explainers()
        
        # Generate attributions for first model only
        if self.models:
            first_model = list(self.models.keys())[0]
            print(f"\n🎯 Running demo with {first_model}")
            
            # Token attributions
            self.generate_token_attributions(first_model, n_examples=3, top_k=10)
            
            # Global insights
            self.aggregate_global_insights(first_model, n_samples=30)
        
        print("\n" + "="*70)
        print(" "*20 + "DEMO COMPLETED!")
        print("="*70)
        print(f"\n📁 Check artifacts in: {self.base_dir}/")
        
        return self.results


def main():
    """Main function"""
    poc = NewsExplainabilityPOC(n_categories=20, max_features=2000)
    results = poc.run_quick_demo()
    return results


if __name__ == "__main__":
    results = main()