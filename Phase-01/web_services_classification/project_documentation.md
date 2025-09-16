# Web Services Classification Project
## Complete Documentation & User Guide

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Project Structure](#3-project-structure)
4. [Installation & Setup](#4-installation--setup)
5. [Execution Steps](#5-execution-steps)
6. [Model Implementation](#6-model-implementation)
7. [Evaluation System](#7-evaluation-system)
8. [Visualization & Analysis](#8-visualization--analysis)
9. [Results & Performance](#9-results--performance)
10. [Advanced Usage](#10-advanced-usage)

---

## 1. Project Overview

### Purpose
This project implements a comprehensive machine learning and deep learning pipeline for **web service classification**. The system automatically categorizes web services into predefined categories using both traditional ML algorithms and modern deep learning approaches.

### Key Features
- **Multi-Model Approach**: Supports both ML (Logistic Regression, Random Forest, XGBoost) and DL (BiLSTM) models
- **Dual Feature Extraction**: TF-IDF and SBERT embeddings for text representation
- **Scalable Categories**: Handles 10, 20, 40, and 50 category classifications
- **Comprehensive Evaluation**: Top-K accuracy, confusion matrices, classification reports
- **Rich Visualizations**: Line plots, bar charts, radar plots for performance analysis
- **Automated Pipeline**: End-to-end execution with minimal manual intervention

### Problem Solved
Automatically categorizing web services based on their descriptions, enabling better service discovery, organization, and recommendation systems.

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Input    │──▶│  Preprocessing   │──▶│ Feature Extract │
│                 │    │                 │    │                 │
│ • Raw Dataset   │    │ • Text Cleaning │    │ • TF-IDF        │
│ • CSV Format    │    │ • Tokenization  │    │ • SBERT         │
│ • Descriptions  │    │ • Label Encoding│    │ • Vectorization │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ML Training   │    │   DL Training   │    │   Evaluation    │
│                 │    │                 │    │                 │
│ • LogisticReg   │    │ • BiLSTM        │    │ • Metrics Calc  │
│ • RandomForest  │    │ • Dense Layers  │    │ • Top-K Accuracy│
│ • XGBoost       │    │ • Callbacks     │    │ • Confusion Mtx │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────┬───────────┘                       │
                     ▼                                   ▼
         ┌─────────────────┐                ┌─────────────────┐
         │  Visualization  │◄───────────────┤     Results     │
         │                 │                │                 │
         │ • Line Plots    │                │ • Performance   │
         │ • Bar Charts    │                │ • Comparisons   │
         │ • Radar Plots   │                │ • Best Models   │
         │ • Comparisons   │                │ • Reports       │
         └─────────────────┘                └─────────────────┘
```

### Data Flow
1. **Input**: Raw web service descriptions and categories
2. **Preprocessing**: Text cleaning, tokenization, category encoding
3. **Feature Extraction**: Convert text to numerical features (TF-IDF/SBERT)
4. **Model Training**: Train multiple ML and DL models simultaneously
5. **Evaluation**: Calculate performance metrics and generate reports
6. **Visualization**: Create comprehensive analysis charts and comparisons

---

## 3. Project Structure

```
web_services_classification/
├── main.py                      # Main pipeline orchestrator
├── requirements.txt             # Python dependencies
├── README.md                   # Basic project information
├── 
├── src/                        # Source code directory
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── 
│   ├── preprocessing/         # Data preprocessing modules
│   │   ├── data_analysis.py   # Dataset analysis and statistics
│   │   ├── data_preprocessing.py  # Data cleaning and preparation
│   │   └── feature_extraction.py # TF-IDF and SBERT extraction
│   │   
│   ├── modeling/              # Model implementations
│   │   ├── ml_models.py       # ML models (LR, RF, XGB)
│   │   ├── dl_models.py       # DL models (BiLSTM)
│   │   ├── evaluate.py        # Common evaluation functions
│   │   └── overall_comparison.py # ML vs DL comparisons
│   │   
│   └── utils/                 # Utility functions
│       ├── utils.py          # General utilities
│       └── logger.py         # Logging configuration
│
├── data/                      # Data directory
│   ├── raw/                  # Original datasets
│   ├── processed/            # Cleaned and processed data
│   ├── splits/               # Train/validation/test splits
│   │   ├── top_10_categories/
│   │   ├── top_20_categories/
│   │   ├── top_40_categories/
│   │   └── top_50_categories/
│   └── features/             # Extracted features
│       ├── tfidf/
│       └── sbert/
│
├── models/                   # Trained model storage
│   ├── ml_models/           # ML model files (.pkl)
│   ├── dl_models/           # DL model files (.h5)
│   └── vectorizers/         # Feature extractors
│
├── results/                 # Analysis results and visualizations
│   ├── ml/                  # ML model results
│   │   ├── comparisons/     # ML comparison charts
│   │   ├── top_10_categories/  # Individual category results
│   │   ├── top_20_categories/
│   │   ├── top_40_categories/
│   │   └── top_50_categories/
│   ├── dl/                  # DL model results
│   │   └── [similar structure]
│   └── overall/             # Combined ML/DL comparisons
│       ├── Overall_Comparison_*.png
│       ├── Overall_TopK_Comparison.png
│       ├── Overall_Bar_Comparison_*.png
│       ├── Overall_radar_*.png
│       └── Overall_Performance_Summary.csv
│
└── logs/                    # Execution logs
    ├── data_analysis.log
    ├── preprocessing.log
    ├── training.log
    └── evaluation.log
```

---

## 4. Installation & Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for DL training)
- 8GB+ RAM recommended
- 10GB+ free disk space

### Installation Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd web_services_classification

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install additional packages for SBERT
pip install sentence-transformers

# 6. Verify installation
python -c "import tensorflow; print('TensorFlow version:', tensorflow.__version__)"
python -c "import sentence_transformers; print('SBERT installed successfully')"
```

### Configuration
Edit `src/config.py` to customize:
- Data paths
- Model parameters
- Training configurations
- Output directories

---

## 5. Execution Steps

### Quick Start (Complete Pipeline)

```bash
# Run the entire pipeline with default settings
python main.py --phase all
```

### Step-by-Step Execution

#### Phase 1: Data Analysis
```bash
python main.py --phase analysis
```
**Output**: Dataset statistics, category distributions, text analysis

#### Phase 2: Data Preprocessing
```bash
python main.py --phase preprocessing
```
**Output**: Cleaned data, train/val/test splits for each category size

#### Phase 3: Feature Extraction
```bash
python main.py --phase features
```
**Output**: TF-IDF vectors, SBERT embeddings for all splits

#### Phase 4: ML Model Training
```bash
python main.py --phase ml_training
```
**Output**: 
- Trained ML models (LogisticRegression, RandomForest, XGBoost)
- Performance metrics and visualizations
- Classification reports and confusion matrices

#### Phase 5: DL Model Training
```bash
python main.py --phase dl_training
```
**Output**:
- Trained DL models (BiLSTM)
- Training history plots
- Performance metrics and visualizations

#### Phase 6: Evaluation
```bash
python main.py --phase evaluation
```
**Output**: Comprehensive model analysis and comparison charts

#### Phase 7: Overall Visualization
```bash
python main.py --phase visualize
```
**Output**: Combined ML vs DL comparison visualizations in `results/overall/`

### Advanced Usage

```bash
# Run with specific categories only
python main.py --phase all --categories 10 20

# Run with verbose logging
python main.py --phase ml_training --verbose

# Run individual phases for debugging
python main.py --phase preprocessing --verbose
```

---

## 6. Model Implementation

### Machine Learning Models

#### 1. Logistic Regression
```python
# Configuration in config.py
"logistic_regression": {
    "max_iter": 1000,
    "random_state": 42,
    "solver": "liblinear",
    "C": 1.0
}
```

#### 2. Random Forest
```python
"random_forest": {
    "n_estimators": 100,
    "random_state": 42,
    "max_depth": None,
    "min_samples_split": 2
}
```

#### 3. XGBoost
```python
"xgboost": {
    "n_estimators": 100,
    "random_state": 42,
    "learning_rate": 0.1,
    "max_depth": 6
}
```

### Deep Learning Models

#### BiLSTM Architecture
- **Input Layer**: Feature dimension (TF-IDF: ~10K, SBERT: 384)
- **Dense Layers**: 512 → 256 → 128 → n_categories
- **Activation**: ReLU (hidden), Softmax (output)
- **Regularization**: Dropout (0.3)
- **Optimizer**: Adam (lr=0.001)
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

### Feature Extraction Methods

#### TF-IDF
- **Max Features**: 10,000
- **N-grams**: (1, 2)
- **Min/Max DF**: 2, 0.95
- **Stop Words**: English

#### SBERT
- **Model**: all-MiniLM-L6-v2
- **Dimension**: 384
- **Normalization**: L2
- **Batch Processing**: 32 samples

---

## 7. Evaluation System

### Metrics Calculated

#### Standard Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

#### Top-K Accuracy
- **Top-1**: Standard accuracy
- **Top-3**: Correct label in top 3 predictions
- **Top-5**: Correct label in top 5 predictions

#### Performance Metrics
- **Training Time**: Model training duration
- **Inference Time**: Prediction time per sample

### Evaluation Outputs

#### Per-Category Results
- Classification reports with category names
- Confusion matrices (heatmaps)
- Performance metrics (CSV format)

#### Comparison Charts
- Line plots: Performance vs category size
- Bar plots: Model comparisons per category
- Radar plots: Multi-metric visualization
- Top-K accuracy trends

---

## 8. Visualization & Analysis

### Individual Model Analysis

#### ML Models (`results/ml/`)
- **Line Plots**: `ML_Model_Performance_*.png`
- **Bar Plots**: `ML_Model_Performance_enhanced_top_*.png`
- **Radar Plots**: `ML_radar_*_top_*_categories.png`
- **Summary**: `ML_Model_Performance_summary.csv`

#### DL Models (`results/dl/`)
- **Line Plots**: `DL_Model_Performance_*.png`
- **Training History**: `BiLSTM_*_top_*_categories_history.png`
- **Bar Plots**: `DL_Model_Performance_enhanced_top_*.png`
- **Radar Plots**: `DL_radar_*_top_*_categories.png`

### Overall Comparison (`results/overall/`)

#### Combined Analysis Files
1. **`Overall_Comparison_accuracy.png`**: Line plot comparing all models
2. **`Overall_TopK_Comparison.png`**: Top-1/3/5 accuracy comparison
3. **`Overall_Bar_Comparison_top_*.png`**: Side-by-side model comparisons
4. **`Overall_radar_*_top_*_categories.png`**: Combined radar charts
5. **`Overall_Performance_Summary.csv`**: Complete results table

### Interpretation Guide

#### Reading Line Plots
- **X-axis**: Number of categories (10, 20, 40, 50)
- **Y-axis**: Performance metric value
- **Lines**: Different model-feature combinations
- **Trends**: How performance changes with complexity

#### Reading Bar Plots
- **Groups**: Different models
- **Colors**: ML (blue tones) vs DL (red tones)
- **Height**: Metric value
- **Comparisons**: Direct model-to-model performance

#### Reading Radar Plots
- **Axes**: Different categories
- **Distance from center**: Performance level
- **Polygons**: Model performance profiles
- **Overlaps**: Similar performance areas

---

## 9. Results & Performance

### Expected Performance Ranges

#### ML Models
- **Accuracy**: 65-85% (varies by category count)
- **Top-3 Accuracy**: 80-95%
- **Top-5 Accuracy**: 85-98%
- **Training Time**: 1-10 minutes
- **Best Feature**: Typically SBERT > TF-IDF

#### DL Models
- **Accuracy**: 70-88% (varies by category count)
- **Top-3 Accuracy**: 85-96%
- **Top-5 Accuracy**: 90-99%
- **Training Time**: 10-60 minutes
- **Best Feature**: SBERT generally superior

### Performance Patterns

#### Category Size Impact
- **10 Categories**: Highest accuracy (80-88%)
- **20 Categories**: Good accuracy (75-85%)
- **40 Categories**: Moderate accuracy (70-80%)
- **50 Categories**: Challenging accuracy (65-78%)

#### Feature Type Comparison
- **SBERT**: Better semantic understanding, higher accuracy
- **TF-IDF**: Faster processing, good baseline performance
- **Recommendation**: Use SBERT for production, TF-IDF for rapid prototyping

### Model Selection Guidelines

#### Choose ML Models When:
- Fast training/inference required
- Interpretability important
- Limited computational resources
- Baseline performance acceptable

#### Choose DL Models When:
- Maximum accuracy required
- Complex text patterns present
- Sufficient training data available
- Computational resources adequate

---

## 10. Advanced Usage

### Custom Category Sizes
```bash
# Train on specific category sizes
python main.py --phase all --categories 15 25
```

### Hyperparameter Tuning
Edit `src/config.py`:
```python
ML_CONFIG = {
    "logistic_regression": {
        "C": [0.1, 1.0, 10.0],  # Grid search values
        "max_iter": 2000
    }
}
```

### Adding New Models
1. Implement in `src/modeling/ml_models.py` or `src/modeling/dl_models.py`
2. Update configuration in `src/config.py`
3. Add to model creation methods

### Custom Visualizations
```python
from src.modeling.evaluate import ModelEvaluator

evaluator = ModelEvaluator()
evaluator.generate_radar_plots("ml", show_plots=True)
```

### Debugging and Troubleshooting

#### Common Issues
1. **Memory Errors**: Reduce batch size in DL config
2. **CUDA Errors**: Check GPU compatibility
3. **File Not Found**: Verify data paths in config
4. **Import Errors**: Check virtual environment activation

#### Log Analysis
Check log files in `logs/` directory for detailed error information:
```bash
tail -f logs/training.log
```

### Performance Optimization
1. **Use GPU**: Enable CUDA for DL training
2. **Parallel Processing**: Increase n_jobs in ML models
3. **Memory Management**: Use smaller batch sizes for large datasets
4. **Feature Caching**: Enable feature caching for repeated experiments

---

## Conclusion

This comprehensive web services classification system provides:

- **Flexibility**: Multiple models and feature types
- **Scalability**: Handles various category sizes
- **Automation**: End-to-end pipeline execution
- **Analysis**: Rich visualizations and comparisons
- **Reproducibility**: Consistent results with proper seeding

The system enables researchers and practitioners to:
- Compare ML vs DL approaches systematically
- Identify optimal model-feature combinations
- Analyze performance across different complexity levels
- Generate publication-ready visualizations and reports

For support or advanced customization, refer to the source code documentation and configuration files.