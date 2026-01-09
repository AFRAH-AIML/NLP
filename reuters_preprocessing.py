"""
Reuters Dataset Preprocessing

The Reuters dataset contains a large collection of news documents 
used for text categorization research.
"""

import nltk
from nltk.corpus import reuters
from text_preprocessing import TextPreprocessor
from typing import Dict, List
import pandas as pd

# Download Reuters corpus if not available
try:
    nltk.data.find('corpora/reuters')
except LookupError:
    nltk.download('reuters')


class ReutersProcessor:
    """Process Reuters dataset with preprocessing techniques"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.fileids = reuters.fileids()
        self.categories = reuters.categories()
    
    def get_document_text(self, fileid: str) -> str:
        """Get text for a specific document"""
        return ' '.join(reuters.words(fileid))
    
    def get_corpus_text(self) -> str:
        """Get full corpus text"""
        return ' '.join(reuters.words())
    
    def get_category_text(self, category: str) -> str:
        """Get text for specific category"""
        fileids = reuters.fileids(categories=category)
        return ' '.join([self.get_document_text(fid) for fid in fileids])
    
    def get_documents_by_category(self, category: str) -> List[str]:
        """Get all documents in a category"""
        fileids = reuters.fileids(categories=category)
        return [self.get_document_text(fid) for fid in fileids]
    
    def preprocess_corpus(self, apply_stemming: bool = False,
                         apply_lemmatization: bool = True) -> List[List[str]]:
        """
        Preprocess entire Reuters corpus
        
        Args:
            apply_stemming: Whether to apply stemming
            apply_lemmatization: Whether to apply lemmatization
            
        Returns:
            List of preprocessed token lists (one per document)
        """
        processed_docs = []
        
        for idx, fileid in enumerate(self.fileids):
            if idx % 1000 == 0:
                print(f"Processing document {idx}/{len(self.fileids)}")
            
            text = self.get_document_text(fileid)
            processed_tokens = self.preprocessor.preprocess_complete(
                text,
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
            Dictionary with category as key and list of preprocessed docs as value
        """
        results = {}
        
        for category in self.categories:
            print(f"Processing category: {category}")
            docs = self.get_documents_by_category(category)
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
        text = self.get_corpus_text()
        
        # Step-by-step preprocessing
        normalized = self.preprocessor.normalize_text(text)
        tokens = self.preprocessor.tokenize(normalized)
        tokens_no_stops = self.preprocessor.remove_stopwords(tokens)
        stemmed = self.preprocessor.stem(tokens_no_stops)
        lemmatized = self.preprocessor.lemmatize(tokens_no_stops)
        
        return {
            'dataset': 'Reuters',
            'total_documents': len(self.fileids),
            'total_categories': len(self.categories),
            'original_char_count': len(text),
            'normalized_char_count': len(normalized),
            'token_count': len(tokens),
            'tokens_after_stopword_removal': len(tokens_no_stops),
            'unique_tokens': len(set(tokens_no_stops)),
            'unique_stemmed': len(set(stemmed)),
            'unique_lemmatized': len(set(lemmatized)),
            'stopword_reduction_percent': (1 - len(tokens_no_stops) / len(tokens)) * 100 if tokens else 0,
            'avg_tokens_per_document': len(tokens) / len(self.fileids) if self.fileids else 0,
        }
    
    def get_category_statistics(self) -> pd.DataFrame:
        """Get preprocessing statistics per category"""
        results = []
        
        for category in self.categories:
            docs = self.get_documents_by_category(category)
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
    
    def get_document_statistics(self) -> pd.DataFrame:
        """Get preprocessing statistics per document"""
        results = []
        
        for idx, fileid in enumerate(self.fileids):
            if idx % 1000 == 0:
                print(f"Processing document {idx}/{len(self.fileids)}")
            
            text = self.get_document_text(fileid)
            normalized = self.preprocessor.normalize_text(text)
            tokens = self.preprocessor.tokenize(normalized)
            tokens_no_stops = self.preprocessor.remove_stopwords(tokens)
            
            results.append({
                'fileid': fileid,
                'token_count': len(tokens),
                'tokens_after_stopword_removal': len(tokens_no_stops),
                'unique_tokens': len(set(tokens_no_stops)),
            })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    processor = ReutersProcessor()
    
    print("\n=== Reuters Dataset Preprocessing ===\n")
    print(f"Total documents: {len(processor.fileids)}")
    print(f"Total categories: {len(processor.categories)}")
    print(f"Categories: {processor.categories[:10]}...\n")
    
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
    cat_stats.to_csv('reuters_category_statistics.csv', index=False)
    print("\n\nCategory statistics saved to 'reuters_category_statistics.csv'")
