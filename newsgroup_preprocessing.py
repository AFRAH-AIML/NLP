"""
20 Newsgroups Dataset Preprocessing

The 20 newsgroups dataset comprises approximately 18,000 newsgroup posts 
on 20 different topics.
"""

from sklearn.datasets import fetch_20newsgroups
from text_preprocessing import TextPreprocessor
from typing import Dict, List
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class NewsgroupProcessor:
    """Process 20 Newsgroups dataset with preprocessing techniques"""
    
    def __init__(self, remove_headers_footers: bool = True, remove_quotes: bool = True):
        """
        Initialize newsgroup processor
        
        Args:
            remove_headers_footers: Remove email headers and footers
            remove_quotes: Remove quoted text
        """
        self.preprocessor = TextPreprocessor()
        self.remove_headers = remove_headers_footers
        self.remove_quotes = remove_quotes
        self.newsgroups = None
    
    def download_dataset(self) -> Dict:
        """Download 20 Newsgroups dataset"""
        print("Downloading 20 Newsgroups dataset...")
        self.newsgroups = fetch_20newsgroups(
            subset='all',
            remove=('headers', 'footers', 'quotes') if self.remove_headers else ()
        )
        print(f"Downloaded {len(self.newsgroups.data)} documents")
        return self.newsgroups
    
    def get_documents(self) -> List[str]:
        """Get all documents"""
        if self.newsgroups is None:
            self.download_dataset()
        return self.newsgroups.data
    
    def get_categories(self) -> List[str]:
        """Get all category names"""
        if self.newsgroups is None:
            self.download_dataset()
        return self.newsgroups.target_names
    
    def get_category_documents(self, category_idx: int) -> List[str]:
        """Get documents for specific category"""
        if self.newsgroups is None:
            self.download_dataset()
        
        docs = []
        for doc, label in zip(self.newsgroups.data, self.newsgroups.target):
            if label == category_idx:
                docs.append(doc)
        return docs
    
    def preprocess_dataset(self, apply_stemming: bool = False,
                          apply_lemmatization: bool = True) -> List[List[str]]:
        """
        Preprocess entire 20 Newsgroups dataset
        
        Args:
            apply_stemming: Whether to apply stemming
            apply_lemmatization: Whether to apply lemmatization
            
        Returns:
            List of preprocessed token lists
        """
        documents = self.get_documents()
        processed_docs = []
        
        for idx, doc in enumerate(documents):
            if idx % 1000 == 0:
                print(f"Processing document {idx}/{len(documents)}")
            
            processed_tokens = self.preprocessor.preprocess_complete(
                doc,
                apply_stemming=apply_stemming,
                apply_lemmatization=apply_lemmatization
            )
            processed_docs.append(processed_tokens)
        
        return processed_docs
    
    def preprocess_by_category(self, apply_stemming: bool = False,
                              apply_lemmatization: bool = True) -> Dict[str, List[List[str]]]:
        """
        Preprocess documents grouped by category
        
        Args:
            apply_stemming: Whether to apply stemming
            apply_lemmatization: Whether to apply lemmatization
            
        Returns:
            Dictionary with category name as key and list of preprocessed docs as value
        """
        categories = self.get_categories()
        results = {}
        
        for cat_idx, category in enumerate(categories):
            print(f"Processing category: {category}")
            docs = self.get_category_documents(cat_idx)
            processed_category_docs = []
            
            for doc in docs:
                processed_tokens = self.preprocessor.preprocess_complete(
                    doc,
                    apply_stemming=apply_stemming,
                    apply_lemmatization=apply_lemmatization
                )
                processed_category_docs.append(processed_tokens)
            
            results[category] = processed_category_docs
        
        return results
    
    def get_preprocessing_summary(self) -> Dict:
        """Get preprocessing summary for entire dataset"""
        documents = self.get_documents()
        full_text = ' '.join(documents)
        
        # Step-by-step preprocessing
        normalized = self.preprocessor.normalize_text(full_text)
        tokens = self.preprocessor.tokenize(normalized)
        tokens_no_stops = self.preprocessor.remove_stopwords(tokens)
        stemmed = self.preprocessor.stem(tokens_no_stops)
        lemmatized = self.preprocessor.lemmatize(tokens_no_stops)
        
        return {
            'dataset': '20 Newsgroups',
            'total_documents': len(documents),
            'total_categories': len(self.get_categories()),
            'original_char_count': len(full_text),
            'normalized_char_count': len(normalized),
            'token_count': len(tokens),
            'tokens_after_stopword_removal': len(tokens_no_stops),
            'unique_tokens': len(set(tokens_no_stops)),
            'unique_stemmed': len(set(stemmed)),
            'unique_lemmatized': len(set(lemmatized)),
            'stopword_reduction_percent': (1 - len(tokens_no_stops) / len(tokens)) * 100 if tokens else 0,
            'avg_tokens_per_document': len(tokens) / len(documents) if documents else 0,
        }
    
    def get_category_statistics(self) -> pd.DataFrame:
        """Get preprocessing statistics per category"""
        categories = self.get_categories()
        results = []
        
        for cat_idx, category in enumerate(categories):
            docs = self.get_category_documents(cat_idx)
            full_text = ' '.join(docs)
            
            normalized = self.preprocessor.normalize_text(full_text)
            tokens = self.preprocessor.tokenize(normalized)
            tokens_no_stops = self.preprocessor.remove_stopwords(tokens)
            stemmed = self.preprocessor.stem(tokens_no_stops)
            lemmatized = self.preprocessor.lemmatize(tokens_no_stops)
            
            results.append({
                'category': category,
                'document_count': len(docs),
                'token_count': len(tokens),
                'tokens_after_stopword_removal': len(tokens_no_stops),
                'unique_tokens': len(set(tokens_no_stops)),
                'unique_stemmed': len(set(stemmed)),
                'unique_lemmatized': len(set(lemmatized)),
                'avg_tokens_per_doc': len(tokens) / len(docs) if docs else 0,
            })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    processor = NewsgroupProcessor()
    
    print("\n=== 20 Newsgroups Dataset Preprocessing ===\n")
    
    # Download dataset
    newsgroups = processor.download_dataset()
    print(f"Categories: {len(processor.get_categories())}\n")
    
    # Get summary
    print("Generating preprocessing summary...\n")
    summary = processor.get_preprocessing_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Get category statistics
    print("\n\nGenerating category statistics...\n")
    cat_stats = processor.get_category_statistics()
    print(cat_stats)
    
    # Save statistics
    cat_stats.to_csv('newsgroups_category_statistics.csv', index=False)
    print("\n\nStatistics saved to 'newsgroups_category_statistics.csv'")
