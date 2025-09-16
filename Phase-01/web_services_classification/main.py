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
#from src.modeling.ml_models import MLModelTrainer
#from src.modeling.dl_model import DLModelTrainer
#from src.evaluation.evaluation import ModelEvaluator
#from src.evaluation.topk_evaluation import TopKEvaluator
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
    """Run ML model training phase"""
    logger = logging.getLogger(__name__)
    logger.info("Starting ML Training Phase")
    
    ml_trainer = MLModelTrainer()
    
    # Train all ML models for all category sizes
    ml_trainer.train_all_categories()
    ml_trainer.plot_ml_results_only()
    
    logger.info("ML Training Phase completed")
    return

def run_dl_training_phase():
    """Run DL model training phase"""
    logger = logging.getLogger(__name__)
    logger.info("Starting DL Training Phase")
    
    dl_trainer = DLModelTrainer()
    
    # Train all DL models for all category sizes
    logger.info(f"Training DL models")
    results = dl_trainer.train_all_categories()
    logger.info("DL Training Phase completed")
    return results


def run_evaluation_phase():
    """Run model evaluation phase"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Evaluation Phase")
    
    evaluator = ModelEvaluator()
    topk_evaluator = TopKEvaluator()
    
    # Evaluate all models
    for n_categories in CATEGORY_SIZES:
        logger.info(f"Evaluating models for top {n_categories} categories")
        
        # Standard evaluation
        evaluator.evaluate_all_models(n_categories)
        
        # Top-K evaluation
        topk_evaluator.evaluate_all_models(n_categories)
    
    logger.info("Evaluation Phase completed")


def run_benchmark_generation_phase():
    """Run benchmark generation phase"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Benchmark Generation Phase")
    
    benchmark_generator = BenchmarkGenerator()
    
    # Generate comprehensive benchmarks
    benchmark_generator.generate_all_benchmarks()
    
    logger.info("Benchmark Generation Phase completed")


def main():
    """Main function to run the entire pipeline"""
    parser = argparse.ArgumentParser(description="Web Services Classification Pipeline")
    parser.add_argument(
        "--phase", 
        choices=[
            "all", "analysis", "preprocessing", "features", "ml_training", 
            "dl_training", "evaluation", "benchmarks"
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
            run_data_analysis_phase()
            run_preprocessing_phase()
            run_feature_extraction_phase()
            run_ml_training_phase()
            run_dl_training_phase()
            run_evaluation_phase()
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
        elif args.phase == "evaluation":
            run_evaluation_phase()
        elif args.phase == "benchmarks":
            run_benchmark_generation_phase()
        
        logger.info("Pipeline completed successfully")
        print(f"\nPipeline completed successfully!")
        print(f"Check the logs directory for detailed logs.")
        print(f"Check the results directory for outputs.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()