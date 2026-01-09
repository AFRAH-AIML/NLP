"""
Master Script: Complete NLP Preprocessing Pipeline

This script demonstrates and tests all preprocessing techniques on all datasets.
"""

import sys
import pandas as pd
from text_preprocessing import TextPreprocessor
from brown_corpus_preprocessing import BrownCorpusProcessor
from ud_english_treebank_preprocessing import UDEnglishTreebankProcessor
from newsgroup_preprocessing import NewsgroupProcessor
from reuters_preprocessing import ReutersProcessor


def print_header(title: str, level: int = 1):
    """Print formatted header"""
    if level == 1:
        print("\n" + "=" * 100)
        print(f" {title}")
        print("=" * 100)
    else:
        print("\n" + "-" * 100)
        print(f" {title}")
        print("-" * 100)


def main():
    """Main execution"""
    
    print_header("NLP TEXT PREPROCESSING - COMPLETE PIPELINE", level=1)
    
    print("\nThis script demonstrates text preprocessing techniques on 4 major NLP datasets:")
    print("  1. Brown Corpus")
    print("  2. Universal Dependencies English Treebank")
    print("  3. 20 Newsgroups Dataset")
    print("  4. Reuters Dataset")
    
    print("\nPreprocessing steps include:")
    print("  • Text Normalization (lowercase, remove special chars, URLs, HTML)")
    print("  • Tokenization (word and sentence level)")
    print("  • Stopword Removal")
    print("  • Stemming (Porter Stemmer)")
    print("  • Lemmatization (WordNet Lemmatizer)")
    
    # Dictionary to store all summaries
    all_summaries = []
    
    # ==================== 1. BROWN CORPUS ====================
    print_header("1. BROWN CORPUS PREPROCESSING", level=1)
    try:
        brown_processor = BrownCorpusProcessor()
        print(f"\nProcessing {len(brown_processor.categories)} categories...")
        print(f"Categories: {brown_processor.categories}")
        
        # Get report
        brown_report = brown_processor.generate_preprocessing_report()
        print("\nPreprocessing Report:")
        print(brown_report.to_string(index=False))
        
        # Save report
        brown_report.to_csv('/workspaces/NLP/reports/brown_corpus_report.csv', index=False)
        print("\n✓ Brown Corpus report saved to: reports/brown_corpus_report.csv")
        
        # Add summary to all summaries
        brown_summary = {
            'Dataset': 'Brown Corpus',
            'Total Documents': len(brown_processor.categories),
            'Tokens': brown_report['token_count'].sum(),
            'Unique Tokens': brown_report['unique_tokens'].sum(),
            'Avg Stopword Reduction %': brown_report['stopword_reduction_percent'].mean(),
        }
        all_summaries.append(brown_summary)
        
    except Exception as e:
        print(f"⚠ Error processing Brown Corpus: {e}")
    
    # ==================== 2. UD ENGLISH TREEBANK ====================
    print_header("2. UNIVERSAL DEPENDENCIES ENGLISH TREEBANK PREPROCESSING", level=1)
    try:
        ud_processor = UDEnglishTreebankProcessor()
        print(f"\nProcessing {len(ud_processor.trees)} sentences...")
        
        # Get summary
        ud_summary_dict = ud_processor.get_preprocessing_summary()
        print("\nPreprocessing Summary:")
        for key, value in ud_summary_dict.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # Get per-sentence statistics
        sentence_stats = ud_processor.preprocess_sentence_by_sentence()
        sentence_stats.to_csv('/workspaces/NLP/reports/ud_english_treebank_stats.csv', index=False)
        print(f"\n✓ Per-sentence statistics saved to: reports/ud_english_treebank_stats.csv")
        print(f"  Total sentences processed: {len(sentence_stats)}")
        
        # Add summary
        all_summaries.append({
            'Dataset': 'UD English Treebank',
            'Total Documents': ud_summary_dict['total_sentences'],
            'Tokens': ud_summary_dict['token_count'],
            'Unique Tokens': ud_summary_dict['unique_tokens'],
            'Avg Stopword Reduction %': ud_summary_dict['stopword_reduction_percent'],
        })
        
    except Exception as e:
        print(f"⚠ Error processing UD English Treebank: {e}")
    
    # ==================== 3. 20 NEWSGROUPS ====================
    print_header("3. 20 NEWSGROUPS DATASET PREPROCESSING", level=1)
    try:
        newsgroup_processor = NewsgroupProcessor()
        newsgroup_data = newsgroup_processor.download_dataset()
        print(f"\nDownloaded {len(newsgroup_processor.get_documents())} documents")
        print(f"Categories: {len(newsgroup_processor.get_categories())}")
        
        # Get summary
        newsgroup_summary_dict = newsgroup_processor.get_preprocessing_summary()
        print("\nPreprocessing Summary:")
        for key, value in newsgroup_summary_dict.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # Get category statistics
        cat_stats = newsgroup_processor.get_category_statistics()
        cat_stats.to_csv('/workspaces/NLP/reports/newsgroups_category_stats.csv', index=False)
        print(f"\n✓ Category statistics saved to: reports/newsgroups_category_stats.csv")
        print("\nTop 5 categories by token count:")
        print(cat_stats.nlargest(5, 'token_count')[['category', 'document_count', 'token_count']])
        
        # Add summary
        all_summaries.append({
            'Dataset': '20 Newsgroups',
            'Total Documents': newsgroup_summary_dict['total_documents'],
            'Tokens': newsgroup_summary_dict['token_count'],
            'Unique Tokens': newsgroup_summary_dict['unique_tokens'],
            'Avg Stopword Reduction %': newsgroup_summary_dict['stopword_reduction_percent'],
        })
        
    except Exception as e:
        print(f"⚠ Error processing 20 Newsgroups: {e}")
    
    # ==================== 4. REUTERS ====================
    print_header("4. REUTERS DATASET PREPROCESSING", level=1)
    try:
        reuters_processor = ReutersProcessor()
        print(f"\nProcessing {len(reuters_processor.fileids)} documents...")
        print(f"Total categories: {len(reuters_processor.categories)}")
        
        # Get summary
        reuters_summary_dict = reuters_processor.get_preprocessing_summary()
        print("\nPreprocessing Summary:")
        for key, value in reuters_summary_dict.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # Get category statistics
        cat_stats = reuters_processor.get_category_statistics()
        cat_stats.to_csv('/workspaces/NLP/reports/reuters_category_stats.csv', index=False)
        print(f"\n✓ Category statistics saved to: reports/reuters_category_stats.csv")
        print("\nTop 5 categories by document count:")
        print(cat_stats.nlargest(5, 'document_count')[['category', 'document_count', 'token_count']])
        
        # Add summary
        all_summaries.append({
            'Dataset': 'Reuters',
            'Total Documents': reuters_summary_dict['total_documents'],
            'Tokens': reuters_summary_dict['token_count'],
            'Unique Tokens': reuters_summary_dict['unique_tokens'],
            'Avg Stopword Reduction %': reuters_summary_dict['stopword_reduction_percent'],
        })
        
    except Exception as e:
        print(f"⚠ Error processing Reuters: {e}")
    
    # ==================== OVERALL COMPARISON ====================
    print_header("OVERALL COMPARISON - ALL DATASETS", level=1)
    
    comparison_df = pd.DataFrame(all_summaries)
    print("\n" + comparison_df.to_string(index=False))
    comparison_df.to_csv('/workspaces/NLP/reports/datasets_comparison.csv', index=False)
    print("\n✓ Comparison saved to: reports/datasets_comparison.csv")
    
    print_header("PREPROCESSING COMPLETE", level=1)
    print("\nGenerated Reports:")
    print("  • reports/brown_corpus_report.csv")
    print("  • reports/ud_english_treebank_stats.csv")
    print("  • reports/newsgroups_category_stats.csv")
    print("  • reports/reuters_category_stats.csv")
    print("  • reports/datasets_comparison.csv")
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
