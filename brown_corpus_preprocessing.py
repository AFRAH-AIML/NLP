"""
Brown Corpus Preprocessing

The Brown Corpus is a balanced corpus of English language consisting of 
about 1 million words of text in 500 samples from 15 categories.
"""

import nltk
from nltk.corpus import brown
from text_preprocessing import TextPreprocessor
from typing import Dict, List
import pandas as pd

# Download Brown corpus if not available
try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown')


class BrownCorpusProcessor:
    """Process Brown Corpus with preprocessing techniques"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.categories = brown.categories()
    
    def get_corpus_text(self) -> str:
        """Get full Brown Corpus text"""
        return ' '.join(brown.words())
    
    def get_category_text(self, category: str) -> str:
        """Get text for specific category"""
        return ' '.join(brown.words(categories=category))
    
    def get_all_categories_text(self) -> Dict[str, str]:
        """Get text for all categories"""
        return {category: self.get_category_text(category) for category in self.categories}
    
    def preprocess_corpus(self, apply_stemming: bool = False, 
                         apply_lemmatization: bool = True) -> Dict[str, List[str]]:
        """
        Preprocess entire Brown Corpus by category
        
        Args:
            apply_stemming: Whether to apply stemming
            apply_lemmatization: Whether to apply lemmatization
            
        Returns:
            Dictionary with preprocessed tokens for each category
        """
        results = {}
        
        for category in self.categories:
            print(f"Processing category: {category}")
            text = self.get_category_text(category)
            processed_tokens = self.preprocessor.preprocess_complete(
                text, 
                apply_stemming=apply_stemming,
                apply_lemmatization=apply_lemmatization
            )
            results[category] = processed_tokens
        
        return results
    
    def get_preprocessing_summary(self, category: str) -> Dict:
        """
        Get detailed preprocessing summary for a category
        
        Args:
            category: Category name
            
        Returns:
            Summary statistics
        """
        text = self.get_category_text(category)
        
        # Step-by-step preprocessing
        normalized = self.preprocessor.normalize_text(text)
        tokens = self.preprocessor.tokenize(normalized)
        tokens_no_stops = self.preprocessor.remove_stopwords(tokens)
        stemmed = self.preprocessor.stem(tokens_no_stops)
        lemmatized = self.preprocessor.lemmatize(tokens_no_stops)
        
        return {
            'category': category,
            'original_char_count': len(text),
            'normalized_char_count': len(normalized),
            'token_count': len(tokens),
            'tokens_after_stopword_removal': len(tokens_no_stops),
            'unique_tokens': len(set(tokens_no_stops)),
            'unique_stemmed': len(set(stemmed)),
            'unique_lemmatized': len(set(lemmatized)),
            'stopword_reduction_percent': (1 - len(tokens_no_stops) / len(tokens)) * 100 if tokens else 0,
        }
    
    def generate_preprocessing_report(self) -> pd.DataFrame:
        """Generate preprocessing report for all categories"""
        summaries = []
        for category in self.categories:
            summary = self.get_preprocessing_summary(category)
            summaries.append(summary)
        
        return pd.DataFrame(summaries)


if __name__ == "__main__":
    processor = BrownCorpusProcessor()
    
    print("\n=== Brown Corpus Preprocessing ===\n")
    print(f"Total categories: {len(processor.categories)}")
    print(f"Categories: {processor.categories}\n")
    
    # Generate report
    print("Generating preprocessing report for all categories...\n")
    report = processor.generate_preprocessing_report()
    print(report)
    
    # Save report
    report.to_csv('brown_corpus_preprocessing_report.csv', index=False)
    print("\nReport saved to 'brown_corpus_preprocessing_report.csv'")
    
    # Example: Preprocess single category
    print("\n=== Example: Processing 'news' category ===\n")
    summary = processor.get_preprocessing_summary('news')
    for key, value in summary.items():
        print(f"{key}: {value}")
