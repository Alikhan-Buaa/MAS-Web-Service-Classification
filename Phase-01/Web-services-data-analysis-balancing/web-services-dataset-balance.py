import pandas as pd
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class RankBasedBalancer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the rank-based data balancer
        """
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.class_stats = {}
        
    def create_balancing_config(self):
        """
        Create the balancing configuration dictionary as specified
        """
        balancing_config = {
            "Rank_1_to_10_Categories.csv": {
                "file_type": "rank_based",
                "current_range": "300-350",
                "method": "undersample",
                "target_samples": 300,
                "description": "Top 10 categories - reduce to 300 samples each"
            },
            "Rank_11_to_20_Categories.csv": {
                "file_type": "rank_based", 
                "current_range": "250-300",
                "method": "oversample",
                "target_samples": 275,
                "description": "Rank 11-20 categories - increase to 275 samples each"
            },
            "Rank_21_to_30_Categories.csv": {
                "file_type": "rank_based",
                "current_range": "200-250", 
                "method": "oversample",
                "target_samples": 250,
                "description": "Rank 21-30 categories - increase to 250 samples each"
            },
            "Rank_31_to_40_Categories.csv": {
                "file_type": "rank_based",
                "current_range": "200-225",
                "method": "oversample", 
                "target_samples": 225,
                "description": "Rank 31-40 categories - increase to 225 samples each"
            },
            "Rank_41_to_50_Categories.csv": {
                "file_type": "rank_based",
                "current_range": "~200",
                "method": "oversample",
                "target_samples": 200, 
                "description": "Rank 41-50 categories - increase to 200 samples each"
            }
        }
        
        return balancing_config
    
    def load_and_analyze_dataset(self, csv_path, text_column='Service Description', class_column='Service Classification'):
        """
        Load dataset and analyze current distribution
        """
        print(f"\nLoading: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"Dataset shape: {df.shape}")
        class_counts = df[class_column].value_counts()
        print(f"Classes: {len(class_counts)}")
        print(f"Sample distribution: Min={class_counts.min()}, Max={class_counts.max()}, Avg={class_counts.mean():.1f}")
        
        return df, class_counts
    
    def calculate_class_statistics(self, df, text_column='Service Description', class_column='Service Classification'):
        """
        Calculate statistics for each class in the dataset
        """
        class_stats = {}
        
        for class_label in df[class_column].unique():
            class_data = df[df[class_column] == class_label]
            
            word_counts = []
            text_lengths = []
            
            for text in class_data[text_column]:
                if pd.isna(text):
                    words = []
                    text_str = ""
                else:
                    text_str = str(text)
                    words = word_tokenize(text_str.lower())
                
                word_counts.append(len(words))
                text_lengths.append(len(text_str))
            
            class_stats[class_label] = {
                'count': len(class_data),
                'avg_words': np.mean(word_counts),
                'std_words': np.std(word_counts),
                'avg_length': np.mean(text_lengths),
                'std_length': np.std(text_lengths),
                'sample_texts': class_data[text_column].tolist()
            }
        
        return class_stats
    
    def undersample_data(self, df, target_samples, text_column='Service Description', class_column='Service Classification'):
        """
        Undersample each class to target_samples
        """
        print(f"Undersampling to {target_samples} samples per class...")
        
        undersampled_data = pd.DataFrame()
        
        for class_label in df[class_column].unique():
            class_data = df[df[class_column] == class_label]
            current_count = len(class_data)
            
            if current_count <= target_samples:
                selected_data = class_data
                print(f"  {class_label}: Keeping all {current_count} samples")
            else:
                selected_data = class_data.sample(n=target_samples, random_state=42)
                print(f"  {class_label}: Reduced from {current_count} to {target_samples}")
            
            undersampled_data = pd.concat([undersampled_data, selected_data], ignore_index=True)
        
        return undersampled_data
    
    def generate_synthetic_text(self, class_label, class_stats, target_words, target_length):
        """
        Generate synthetic text based on class patterns
        """
        sample_texts = class_stats[class_label]['sample_texts']
        
        # Extract vocabulary and patterns
        all_words = []
        common_phrases = []
        
        for text in sample_texts:
            if pd.isna(text):
                continue
            words = word_tokenize(str(text).lower())
            all_words.extend(words)
            
            # Extract 2-word phrases
            for i in range(len(words) - 1):
                if len(words[i]) > 2 and len(words[i+1]) > 2:
                    common_phrases.append(f"{words[i]} {words[i+1]}")
        
        word_freq = Counter(all_words)
        
        # Get common words (excluding stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        common_words = [word for word, count in word_freq.most_common(50) 
                       if word not in stop_words and len(word) > 2]
        
        if not common_words:
            common_words = ['service', 'web', 'application', 'system', 'platform']
        
        # Start with a template
        valid_texts = [text for text in sample_texts if not pd.isna(text)]
        if not valid_texts:
            return "generated service description"
            
        template = random.choice(valid_texts)
        template_words = word_tokenize(str(template).lower())
        
        # Generate synthetic text
        synthetic_words = []
        current_words = 0
        
        for word in template_words:
            if current_words >= target_words:
                break
                
            if word in stop_words or len(word) <= 2:
                synthetic_words.append(word)
            else:
                # 30% chance to replace with similar word from class
                if random.random() < 0.3 and common_words:
                    replacement = random.choice(common_words)
                    synthetic_words.append(replacement)
                else:
                    synthetic_words.append(word)
            
            current_words += 1
        
        # Add more words if needed
        while current_words < target_words and common_words:
            synthetic_words.append(random.choice(common_words))
            current_words += 1
        
        synthetic_text = ' '.join(synthetic_words)
        
        # Adjust length
        if len(synthetic_text) < target_length * 0.8:
            additional_words = random.choices(common_words, k=min(5, len(common_words)))
            synthetic_text += ' ' + ' '.join(additional_words)
        elif len(synthetic_text) > target_length * 1.2:
            synthetic_text = synthetic_text[:int(target_length * 1.1)]
            if ' ' in synthetic_text:
                synthetic_text = synthetic_text.rsplit(' ', 1)[0]
        
        return synthetic_text.strip()
    
    def check_similarity_with_samples(self, new_text, existing_embeddings, similarity_threshold=0.80, num_samples=5):
        """
        Check similarity with 5 random existing samples
        """
        if len(existing_embeddings) == 0:
            return False
            
        new_embedding = self.model.encode([new_text])
        
        # Sample random embeddings for comparison
        num_existing = len(existing_embeddings)
        sample_indices = random.sample(range(num_existing), min(num_samples, num_existing))
        sampled_embeddings = existing_embeddings[sample_indices]
        
        similarities = cosine_similarity(new_embedding, sampled_embeddings)[0]
        max_similarity = np.max(similarities)
        
        return max_similarity >= similarity_threshold
    
    def oversample_data(self, df, target_samples, text_column='Service Description', 
                       class_column='Service Classification', similarity_threshold=0.80):
        """
        Oversample each class to target_samples using synthetic generation
        """
        print(f"Oversampling to {target_samples} samples per class...")
        
        # Calculate class statistics
        class_stats = self.calculate_class_statistics(df, text_column, class_column)
        
        # Generate embeddings for similarity checking
        embeddings = self.model.encode(df[text_column].tolist())
        
        oversampled_data = df.copy()
        
        for class_label in df[class_column].unique():
            current_count = len(df[df[class_column] == class_label])
            needed_samples = target_samples - current_count
            
            if needed_samples <= 0:
                print(f"  {class_label}: Already has {current_count} samples")
                continue
            
            print(f"  {class_label}: Generating {needed_samples} synthetic samples...")
            
            stats = class_stats[class_label]
            generated_count = 0
            failed_attempts = 0
            max_attempts = 100
            
            while generated_count < needed_samples and failed_attempts < max_attempts:
                # Target characteristics (above average)
                target_words = max(5, int(np.random.normal(
                    stats['avg_words'] * 1.1,  # 10% above average
                    stats['std_words'] * 0.3
                )))
                
                target_length = max(20, int(np.random.normal(
                    stats['avg_length'] * 1.1,  # 10% above average
                    stats['std_length'] * 0.3
                )))
                
                # Generate synthetic text
                synthetic_text = self.generate_synthetic_text(
                    class_label, class_stats, target_words, target_length
                )
                
                # Check similarity
                if not self.check_similarity_with_samples(synthetic_text, embeddings, similarity_threshold, 5):
                    # Add to dataset
                    new_row = {
                        text_column: synthetic_text,
                        class_column: class_label
                    }
                    oversampled_data = pd.concat([oversampled_data, pd.DataFrame([new_row])], ignore_index=True)
                    
                    # Update embeddings
                    new_embedding = self.model.encode([synthetic_text])
                    embeddings = np.vstack([embeddings, new_embedding])
                    
                    generated_count += 1
                    failed_attempts = 0
                else:
                    failed_attempts += 1
            
            if failed_attempts >= max_attempts:
                print(f"    Warning: Generated {generated_count}/{needed_samples} samples (similarity limit reached)")
        
        return oversampled_data
    
    def process_rank_based_files(self, input_dir, output_dir, text_column='Service Description', 
                                class_column='Service Classification'):
        """
        Process all rank-based files according to the configuration
        """
        # Get balancing configuration
        config = self.create_balancing_config()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(output_dir, "balancing_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to: {config_path}")
        
        # Process each file
        results_summary = {}
        
        for filename, settings in config.items():
            input_path = os.path.join(input_dir, filename)
            
            if not os.path.exists(input_path):
                print(f"\nWarning: File not found: {input_path}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing: {filename}")
            print(f"Method: {settings['method'].upper()}")
            print(f"Target samples: {settings['target_samples']}")
            print(f"Description: {settings['description']}")
            print(f"{'='*60}")
            
            # Load and analyze dataset
            df, class_counts = self.load_and_analyze_dataset(input_path, text_column, class_column)
            
            # Apply balancing - THIS IS WHERE THE ACTUAL BALANCING HAPPENS
            if settings['method'] == 'undersample':
                print(f"Calling undersample_data() function...")
                balanced_df = self.undersample_data(df, settings['target_samples'], text_column, class_column)
            elif settings['method'] == 'oversample':
                print(f"Calling oversample_data() function...")
                balanced_df = self.oversample_data(df, settings['target_samples'], text_column, class_column, similarity_threshold=0.80)
            else:
                print(f"Unknown method: {settings['method']}")
                continue
            
            # Save balanced dataset
            output_filename = filename.replace('.csv', '_balanced.csv')
            output_path = os.path.join(output_dir, output_filename)
            balanced_df.to_csv(output_path, index=False)
            
            # Analyze results
            balanced_counts = balanced_df[class_column].value_counts()
            
            results_summary[filename] = {
                'original_shape': df.shape,
                'balanced_shape': balanced_df.shape,
                'original_classes': len(class_counts),
                'balanced_classes': len(balanced_counts),
                'original_samples_range': f"{class_counts.min()}-{class_counts.max()}",
                'balanced_samples_range': f"{balanced_counts.min()}-{balanced_counts.max()}",
                'method': settings['method'],
                'target_samples': settings['target_samples']
            }
            
            print(f"\nResults for {filename}:")
            print(f"  Original: {df.shape[0]} samples, {len(class_counts)} classes")
            print(f"  Balanced: {balanced_df.shape[0]} samples, {len(balanced_counts)} classes")
            print(f"  Sample range: {class_counts.min()}-{class_counts.max()} → {balanced_counts.min()}-{balanced_counts.max()}")
            print(f"  Saved to: {output_path}")
        
        # Save results summary
        summary_path = os.path.join(output_dir, "balancing_results_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # FINAL STEP: Concatenate all balanced datasets into final web_services_dataset.csv
        print(f"\n{'='*60}")
        print("CREATING FINAL CONSOLIDATED DATASET")
        print(f"{'='*60}")
        
        final_dataset = self.create_final_dataset(output_dir, text_column, class_column)
        
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETED!")
        print(f"Results summary saved to: {summary_path}")
        print(f"All balanced datasets saved in: {output_dir}")
        print(f"Final consolidated dataset: web_services_dataset.csv")
        print(f"{'='*60}")
        
        return results_summary
    
    def create_final_dataset(self, output_dir, text_column='Service Description', class_column='Service Classification'):
        """
        Concatenate all balanced rank files into final web_services_dataset.csv
        """
        balanced_files = [
            "Rank_1_to_10_Categories_balanced.csv",
            "Rank_11_to_20_Categories_balanced.csv", 
            "Rank_21_to_30_Categories_balanced.csv",
            "Rank_31_to_40_Categories_balanced.csv",
            "Rank_41_to_50_Categories_balanced.csv"
        ]
        
        final_dataset = pd.DataFrame()
        total_samples = 0
        total_classes = set()
        
        print("Concatenating balanced datasets:")
        
        for filename in balanced_files:
            file_path = os.path.join(output_dir, filename)
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"  ✓ {filename}: {df.shape[0]} samples, {len(df[class_column].unique())} classes")
                
                # Add source rank information
                rank_info = filename.replace("_balanced.csv", "").replace("_", " ")
                df['Source_Rank'] = rank_info
                
                final_dataset = pd.concat([final_dataset, df], ignore_index=True)
                total_samples += df.shape[0]
                total_classes.update(df[class_column].unique())
            else:
                print(f"  ✗ {filename}: File not found")
        
        if not final_dataset.empty:
            # Reorder columns to have Service Description and Service Classification first
            columns_order = [text_column, class_column, 'Source_Rank']
            # Add any other columns that might exist
            other_columns = [col for col in final_dataset.columns if col not in columns_order]
            final_columns = columns_order + other_columns
            
            final_dataset = final_dataset[final_columns]
            
            # Save final dataset
            final_path = os.path.join(output_dir, "web_services_dataset.csv")
            final_dataset.to_csv(final_path, index=False)
            
            print(f"\nFinal Dataset Summary:")
            print(f"  Total samples: {total_samples}")
            print(f"  Total classes: {len(total_classes)}")
            print(f"  Columns: {list(final_dataset.columns)}")
            print(f"  Final dataset saved to: {final_path}")
            
            # Create final distribution summary
            class_distribution = final_dataset[class_column].value_counts()
            print(f"\nClass distribution in final dataset:")
            print(f"  Min samples per class: {class_distribution.min()}")
            print(f"  Max samples per class: {class_distribution.max()}")
            print(f"  Average samples per class: {class_distribution.mean():.1f}")
            
            # Save class distribution
            distribution_path = os.path.join(output_dir, "final_class_distribution.csv")
            class_distribution.to_csv(distribution_path, header=['Count'])
            print(f"  Class distribution saved to: {distribution_path}")
            
        else:
            print("Warning: No balanced files found to concatenate!")
            final_dataset = None
        
        return final_dataset

def main():
    """
    Main function to run the rank-based balancing
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Rank-Based Dataset Balancing')
    parser.add_argument('--input_dir', type=str, default='comprehensive_web_services_analysis', 
                       help='Directory containing rank-based CSV files (default: comprehensive_web_services_analysis)')
    parser.add_argument('--output_dir', type=str, default='dataset_balanced',
                       help='Output directory for balanced datasets (default: dataset_balanced)')
    parser.add_argument('--text_column', type=str, default='Service Description',
                       help='Name of text column')
    parser.add_argument('--class_column', type=str, default='Service Classification',
                       help='Name of class column')
    
    args = parser.parse_args()
    
    # Initialize balancer
    balancer = RankBasedBalancer()
    
    # Process all rank-based files
    results = balancer.process_rank_based_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        text_column=args.text_column,
        class_column=args.class_column
    )

if __name__ == "__main__":
    # Example usage when run directly
    import sys
    
    if len(sys.argv) == 1:
        print("Rank-Based Dataset Balancing Script")
        print("="*40)
        print("\nConfiguration:")
        balancer = RankBasedBalancer()
        config = balancer.create_balancing_config()
        
        for filename, settings in config.items():
            print(f"\n{filename}:")
            print(f"  Method: {settings['method']}")
            print(f"  Target: {settings['target_samples']} samples")
            print(f"  Range: {settings['current_range']}")
        
        print("\nUsing default directories:")
        print("  Input: data/category-wise")
        print("  Output: data/balanced/")
        print("\nStarting balancing process...")
        
        # Run with default settings
        results = balancer.process_rank_based_files(
            input_dir='data/analysis/category-wise',
            output_dir='data/balanced',
            text_column='Service Description',
            class_column='Service Classification'
        )
    else:
        main()