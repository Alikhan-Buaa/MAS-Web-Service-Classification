# MAS-Web-Service-Classification

## Phase-01

1. Objective: Establish baseline performance for web service classification.
2. Dataset: Top-N categories (10, 20, 30, 40, 50).
3. Models: ML (Logistic Regression, Random Forest, XGBoost) and DL (BiLSTM).
4. Features: TF-IDF and SBERT embeddings.
5. Evaluation: Top-1, Top-3, Top-5 accuracy, Macro/Micro F1, confusion matrices.
6. Balanced datasets with fixed 80/10/10 train/val/test splits.
7. Reproducible configurations stored in YAML files.
8. Benchmarking: Leaderboards, Top-K curves, and confusion matrices.
9. Total models trained: 40 (30 ML + 10 DL).
10. Analysis includes cross-model comparison, ranking quality, and category difficulty.

## Phase-04

1. **Objective:** Establish baseline performance for web service classification.
2. **Dataset:** Top-N categories (50).
3. **Models:** ML (Logistic Regression, Random Forest, XGBoost), DL (BiLSTM), RoBERTa (Small & Large), Fusion (DeepSeek + RoBERTa + Classifier).
4. **Features:** TF-IDF and SBERT embeddings.
5. **Evaluation:** Top-1, Top-3, Top-5 accuracy, Macro/Micro F1, and confusion matrices.
6. **Balanced Datasets:** Fixed 80/10/10 train/validation/test splits.
7. **Reproducibility:** Configurations stored in YAML files.
8. **Benchmarking:** Leaderboards, Top-K curves, and confusion matrices.
9. **Total Models Trained:** 40 (30 ML + 10 DL).
10. **Analysis:** Cross-model comparison, ranking quality, and category difficulty.


##  Steps for Phase-04

```bash
# 1️ Clone the repository
git clone git@github.com:Alikhan-Buaa/MAS-Web-Service-Classification.git

# 2️ Navigate to the Phase-04 directory
cd MAS-Web-Service-Classification/Phase-04/web_services_classification/

# 3️ Install dependencies
pip install -r ./requirements.txt

# 4️ Download necessary NLTK resources
python -m nltk.downloader punkt stopwords wordnet

# 5️ Run each phase step-by-step

# 6 Data Analysis Phase
python ./main.py --phase analysis

# 7 Preprocessing Phase
python ./main.py --phase preprocessing

# 8 Feature Extraction Phase
python ./main.py --phase features

# 9 Machine Learning Training Phase
python ./main.py --phase ml_training

# 10 Deep Learning Training Phase
python ./main.py --phase dl_training

# 11 BERT Training Phase
python ./main.py --phase bert_training

# 12 Fusion Model Training Phase
python ./main.py --phase fusion_training

# 13 Evaluation Phase
python ./main.py --phase evaluation

# 14 Visualization Phase
python ./main.py --phase visualize
