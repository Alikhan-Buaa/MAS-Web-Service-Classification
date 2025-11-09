# XAI-Explainability
XAI-Explainability

# Step 1: Create Vitual Env and Install the required
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Step 2:

(base) D:\Projects\AI_Expalinability>PYTHON ./ML-Explainability.py
INFO:__main__:Created output directories under xai_artifacts

======================================================================
               20-CATEGORY EXPLAINABILITY DEMO
======================================================================

============================================================
LOADING 20 NEWSGROUPS DATASET
============================================================

Loading 20 categories...
✓ Loaded 20 categories
✓ Training samples: 11314
✓ Test samples: 7532
✓ Features (words): 2000

============================================================
TRAINING MODELS
============================================================

Training LogisticRegression...
✓ LogisticRegression - Accuracy: 0.592

Training RandomForest...
✓ RandomForest - Accuracy: 0.482

Training XGBoost...
✓ XGBoost - Accuracy: 0.544

============================================================
SETTING UP EXPLAINERS
============================================================

Setting up explainers for LogisticRegression...
✓ LogisticRegression explainers ready

Setting up explainers for RandomForest...
✓ RandomForest explainers ready

Setting up explainers for XGBoost...
✓ XGBoost explainers ready

🎯 Running demo with LogisticRegression

============================================================
GENERATING TOKEN ATTRIBUTIONS
============================================================

--- Example 1/3 ---
True: rec.autos | Predicted: rec.autos (21.96%)

--- Example 2/3 ---
True: comp.windows.x | Predicted: rec.motorcycles (22.63%)

--- Example 3/3 ---
True: alt.atheism | Predicted: talk.religion.misc (13.11%)

✓ Saved LIME attributions to xai_artifacts\lime\lime_LogisticRegression_attributions.json
✓ Saved SHAP attributions to xai_artifacts\shap\shap_LogisticRegression_attributions.json

============================================================
AGGREGATING GLOBAL INSIGHTS
============================================================

📊 Computing Global Feature Importance...

🌐 TOP 10 MOST INFLUENTIAL FEATURES:
 1. 'window': 0.0222
 2. 'jesus': 0.0196
 3. 'car': 0.0164
 4. 'sale': 0.0136
 5. 'bike': 0.0130
 6. 'card': 0.0106
 7. 'use': 0.0101
 8. 'god': 0.0098
 9. 'dos': 0.0088
10. 'people': 0.0087

📈 Computing Per-Class Feature Importance...
✓ Saved visualization to xai_artifacts\visualizations\global_insights_LogisticRegression_simple.png

======================================================================
                    DEMO COMPLETED!
======================================================================

📁 Check artifacts in: xai_artifacts/
