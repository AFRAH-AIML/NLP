"""
Universal Dependencies English Treebank Preprocessing

The UD English Web Treebank is a syntactically-annotated corpus 
of English that follows the Universal Dependencies guidelines.
"""

import nltk
from nltk.corpus import dependency_treebank
from text_preprocessing import TextPreprocessor
from typing import Dict, List
import pandas as pd

# Download UD Treebank if not available
try:
    nltk.data.find('corpora/dependency_treebank')
except LookupError:
    nltk.download('dependency_treebank')


class UDEnglishTreebankProcessor:
    """Process Universal Dependencies English Treebank with preprocessing techniques"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.trees = dependency_treebank.parsed_sents()
    
    def get_corpus_text(self) -> str:
        """Get full corpus text from dependency trees"""
        words = []
        for tree in self.trees:
            words.extend([node['word'] for node in tree.nodes.values() if node['word'] is not None])
        return ' '.join(words)
    
    def get_sentence_from_tree(self, tree) -> str:
        """Extract sentence text from dependency tree"""
        words = []
        for i in sorted(tree.nodes.keys()):
            if i == 0:  # Skip root node
                continue
            words.append(tree.nodes[i]['word'])
        return ' '.join(words)
    
    def extract_all_sentences(self) -> List[str]:
        """Extract all sentences from dependency trees"""
        sentences = []
        for tree in self.trees:
            sentence = self.get_sentence_from_tree(tree)
            if sentence.strip():
                sentences.append(sentence)
        return sentences
    
    def preprocess_treebank(self, apply_stemming: bool = False,
                           apply_lemmatization: bool = True) -> List[List[str]]:
        """
        Preprocess entire treebank
        
        Args:
            apply_stemming: Whether to apply stemming
            apply_lemmatization: Whether to apply lemmatization
            
        Returns:
            List of preprocessed token lists (one per sentence)
        """
        sentences = self.extract_all_sentences()
        processed_sentences = []
        
        for idx, sentence in enumerate(sentences):
            if idx % 100 == 0:
                print(f"Processing sentence {idx}/{len(sentences)}")
            
            processed_tokens = self.preprocessor.preprocess_complete(
                sentence,
                apply_stemming=apply_stemming,
                apply_lemmatization=apply_lemmatization
            )
            processed_sentences.append(processed_tokens)
        
        return processed_sentences
    
    def get_preprocessing_summary(self) -> Dict:
        """Get preprocessing summary for entire treebank"""
        text = self.get_corpus_text()
        sentences = self.extract_all_sentences()
        
        # Step-by-step preprocessing
        normalized = self.preprocessor.normalize_text(text)
        tokens = self.preprocessor.tokenize(normalized)
        tokens_no_stops = self.preprocessor.remove_stopwords(tokens)
        stemmed = self.preprocessor.stem(tokens_no_stops)
        lemmatized = self.preprocessor.lemmatize(tokens_no_stops)
        
        return {
            'corpus_name': 'UD English Treebank',
            'total_sentences': len(self.trees),
            'extracted_sentences': len(sentences),
            'original_char_count': len(text),
            'normalized_char_count': len(normalized),
            'token_count': len(tokens),
            'tokens_after_stopword_removal': len(tokens_no_stops),
            'unique_tokens': len(set(tokens_no_stops)),
            'unique_stemmed': len(set(stemmed)),
            'unique_lemmatized': len(set(lemmatized)),
            'stopword_reduction_percent': (1 - len(tokens_no_stops) / len(tokens)) * 100 if tokens else 0,
            'avg_sentence_length': len(tokens) / len(sentences) if sentences else 0,
        }
    
    def preprocess_sentence_by_sentence(self, apply_stemming: bool = False,
                                       apply_lemmatization: bool = True) -> pd.DataFrame:
        """
        Preprocess and return statistics for each sentence
        
        Args:
            apply_stemming: Whether to apply stemming
            apply_lemmatization: Whether to apply lemmatization
            
        Returns:
            DataFrame with per-sentence statistics
        """
        sentences = self.extract_all_sentences()
        results = []
        
        for idx, sentence in enumerate(sentences):
            normalized = self.preprocessor.normalize_text(sentence)
            tokens = self.preprocessor.tokenize(normalized)
            tokens_no_stops = self.preprocessor.remove_stopwords(tokens)
            stemmed = self.preprocessor.stem(tokens_no_stops)
            lemmatized = self.preprocessor.lemmatize(tokens_no_stops)
            
            results.append({
                'sentence_id': idx,
                'original_length': len(sentence),
                'token_count': len(tokens),
                'tokens_after_stopword_removal': len(tokens_no_stops),
                'unique_tokens': len(set(tokens_no_stops)),
                'unique_stemmed': len(set(stemmed)),
                'unique_lemmatized': len(set(lemmatized)),
            })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    processor = UDEnglishTreebankProcessor()
    
    print("\n=== UD English Treebank Preprocessing ===\n")
    
    # Get summary
    print("Generating preprocessing summary...\n")
    summary = processor.get_preprocessing_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Get per-sentence statistics
    print("\n\nGenerating per-sentence statistics...\n")
    sentence_stats = processor.preprocess_sentence_by_sentence()
    print(sentence_stats.head(10))
    
    # Save statistics
    sentence_stats.to_csv('ud_english_treebank_statistics.csv', index=False)
    print("\n\nStatistics saved to 'ud_english_treebank_statistics.csv'")
