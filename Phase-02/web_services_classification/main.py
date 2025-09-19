"""
Web Services Classification Project - Main Entry Point
Run all phases of the classification pipeline
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import (
    CATEGORY_SIZES, LOGGING_CONFIG
    #print_config_summary #validate_config
)
from src.preprocessing.data_analysis import DataAnalyzer
from src.preprocessing.data_preprocessing import DataPreprocessor
from src.preprocessing.feature_extraction import FeatureExtractor
from src.modeling.ml_models import MLModelTrainer
from src.modeling.dl_models import DLModelTrainer
from src.evaluation.evaluate import ModelEvaluator
from src.evaluation.overall_comparison import OverallPerformanceAnalyzer
from src.modeling.bert_models import  RoBERTaModelTrainer
#from src.evaluation.benchmark_generator import BenchmarkGenerator
from src.utils.utils import setup_logging


def setup_project_logging():
    """Setup logging for the entire project"""
    setup_logging(
        log_file=LOGGING_CONFIG['log_files']['data_analysis'],
        level=LOGGING_CONFIG['level'],
        format_str=LOGGING_CONFIG['format']
    )

def run_data_analysis_phase():
    """Run data analysis phase"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Data Analysis Phase")
    
    analyzer = DataAnalyzer()
    
    # Perform data analysis
    results = analyzer.run_complete_analysis() 
    logger.info("Data Analysis Phase completed")
    return results


def run_preprocessing_phase():
    """Run preprocessing Phase"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Preprocessing Phase")
    
    preprocessor = DataPreprocessor()
    results = preprocessor.process_all_categories()
        
    logger.info("Data preprocessing completed successfully!")
    return results   


def run_feature_extraction_phase(categories=None, feature_types=None):
    """Run feature extraction"""
    logger = logging.getLogger(__name__)
    logger.info("Starting feature extraction...")
        
    if feature_types is None:
        feature_types = ['tfidf', 'sbert']
        
    extractor = FeatureExtractor()
    results = extractor.extract_features_all_categories(feature_types)
           
    logger.info("Feature extraction completed successfully!")
    return results
        

def run_ml_training_phase():
    """Run ML model training phase with automatic visualization"""
    logger = logging.getLogger(__name__)
    logger.info("Starting ML Training Phase")
    
    ml_trainer = MLModelTrainer()
    
    # Train all ML models for all category sizes (includes automatic visualization)
    results = ml_trainer.train_all_categories()
    
    logger.info("ML Training Phase completed (with visualizations)")
    return results

def run_dl_training_phase():
    """Run DL model training phase with automatic visualization"""
    logger = logging.getLogger(__name__)
    logger.info("Starting DL Training Phase")
    
    dl_trainer = DLModelTrainer()
    
    # Train all DL models for all category sizes (includes automatic visualization)
    results = dl_trainer.train_all_categories()
    
    logger.info("DL Training Phase completed (with visualizations)")
    return results

def run_bert_training_phase():
    """Run BERT model training phase with automatic visualization"""
    logger = logging.getLogger(__name__)
    bert_trainer = RoBERTaModelTrainer()
    results = bert_trainer.train_all_categories()
    logger.info("BERT Training Phase completed (with visualizations)")
    return results

def run_evaluation_phase():
    """Run comprehensive model evaluation and comparison phase"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Evaluation Phase")
    
    evaluator = ModelEvaluator()
    
    try:
        # Generate ML model analysis
        logger.info("Generating ML model analysis...")
        from src.config import RESULTS_CONFIG
        ml_results_file = RESULTS_CONFIG["ml_comparisons_path"] / "ml_final_results.pkl"
        ml_charts_dir = RESULTS_CONFIG["ml_comparisons_path"] / "charts"
        
        if ml_results_file.exists():
            evaluator.plot_results_comparison(ml_results_file, ml_charts_dir, "ml")
            evaluator.generate_radar_plots("ml")
            logger.info("ML model analysis completed")
        else:
            logger.warning("ML results file not found, skipping ML analysis")
        
        # Generate DL model analysis
        logger.info("Generating DL model analysis...")
        dl_results_file = RESULTS_CONFIG["dl_comparisons_path"] / "dl_final_results.pkl"
        dl_charts_dir = RESULTS_CONFIG["dl_comparisons_path"] / "charts"
        
        if dl_results_file.exists():
            evaluator.plot_results_comparison(dl_results_file, dl_charts_dir, "dl")
            evaluator.generate_radar_plots("dl")
            logger.info("DL model analysis completed")
        else:
            logger.warning("DL results file not found, skipping DL analysis")
            
    except Exception as e:
        logger.error(f"Error in evaluation phase: {e}")
        print(f"Warning: Some evaluation steps may have failed: {e}")
    
    logger.info("Evaluation Phase completed")

def run_visualize_phase():
    """Run overall performance visualization phase (ML vs DL comparisons)"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Overall Performance Visualization Phase")
    
    try:
        analyzer = OverallPerformanceAnalyzer()
        analyzer.generate_all_comparisons()
        
        logger.info("Overall Performance Visualization Phase completed successfully")
        
    except Exception as e:
        logger.error(f"Error in overall visualization phase: {e}")
        print(f"Warning: Overall visualization phase encountered errors: {e}")
    
    logger.info("Overall Performance Visualization Phase completed")

def run_benchmark_generation_phase():
    """Run benchmark generation phase"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Benchmark Generation Phase")
    
    # TODO: Implement benchmark generation
    print("Benchmark generation phase - to be implemented")
    print("This will generate:")
    print("- Performance benchmarks across all models")
    print("- Comparison tables and reports")
    print("- Model ranking and recommendations")
    
    logger.info("Benchmark Generation Phase completed")


def main():
    """Main function to run the entire pipeline"""
    parser = argparse.ArgumentParser(description="Web Services Classification Pipeline")
    parser.add_argument(
        "--phase", 
        choices=[
            "all", "analysis", "preprocessing", "features", "ml_training", 
            "dl_training", "bert_training", "evaluation", "visualize", "benchmarks"
        ],
        default="all",
        help="Which phase to run"
    )
    parser.add_argument("--categories", type=int, nargs="+", help="Specific category sizes to process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_project_logging()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Validate configuration
        #validate_config()
        #print_config_summary()
        
        logger.info("Starting Web Services Classification Pipeline")
        logger.info(f"Phase: {args.phase}")
        
        # Override category sizes if specified
        if args.categories:
            global CATEGORY_SIZES
            CATEGORY_SIZES = args.categories
            logger.info(f"Using custom category sizes: {CATEGORY_SIZES}")
        
        # Run specified phase(s)
        if args.phase == "all":
            print("Running complete pipeline...")
            run_data_analysis_phase()
            run_preprocessing_phase()
            run_feature_extraction_phase()
            run_ml_training_phase()
            run_dl_training_phase()
            run_bert_training_phase()
            run_evaluation_phase()
            run_visualize_phase()
            run_benchmark_generation_phase()
        elif args.phase == "analysis":
            run_data_analysis_phase()
        elif args.phase == "preprocessing":
            run_preprocessing_phase()
        elif args.phase == "features":
            run_feature_extraction_phase()
        elif args.phase == "ml_training":
            run_ml_training_phase()
        elif args.phase == "dl_training":
            run_dl_training_phase()
        elif args.phase == "bert_training":
            run_bert_training_phase()
        elif args.phase == "evaluation":
            run_evaluation_phase()
        elif args.phase == "visualize":
            run_visualize_phase()
        elif args.phase == "benchmarks":
            run_benchmark_generation_phase()
        
        logger.info("Pipeline completed successfully")
        print(f"\nPipeline completed successfully!")
        print(f"Check the logs directory for detailed logs.")
        print(f"Check the results directory for outputs:")
        print(f"  - ML results: results/ml/comparisons/")
        print(f"  - DL results: results/dl/comparisons/")
        print(f"  - BERT results: results/bert/comparisons/")
        print(f"  - Individual category results: results/ml|dl|bert/top_X_categories/")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()