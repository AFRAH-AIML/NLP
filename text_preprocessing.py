"""
Text Preprocessing Module for NLP

This module provides text preprocessing functions including:
- Text Normalization
- Tokenization
- Stopword Removal
- Stemming
- Lemmatization
"""

import re
import string
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords, brown
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown')


class TextPreprocessor:
    """Main text preprocessing class"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by:
        - Converting to lowercase
        - Removing special characters and extra whitespace
        - Removing HTML tags and URLs
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str, use_sentence: bool = False) -> List[str]:
        """
        Tokenize text into words or sentences
        
        Args:
            text: Input text
            use_sentence: If True, tokenize into sentences; else tokenize into words
            
        Returns:
            List of tokens (words or sentences)
        """
        if use_sentence:
            return sent_tokenize(text)
        else:
            return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list
        
        Args:
            tokens: List of tokens
            
        Returns:
            Tokens with stopwords removed
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def stem(self, tokens: List[str]) -> List[str]:
        """
        Apply Porter Stemming to tokens
        
        Args:
            tokens: List of tokens
            
        Returns:
            Stemmed tokens
        """
        return [self.stemmer.stem(token.lower()) for token in tokens]
    
    def lemmatize(self, tokens: List[str], pos_tags: List[Tuple] = None) -> List[str]:
        """
        Apply WordNet Lemmatization to tokens
        
        Args:
            tokens: List of tokens
            pos_tags: Optional POS tags for better lemmatization
            
        Returns:
            Lemmatized tokens
        """
        if pos_tags is None:
            # Default to noun if no POS tags provided
            return [self.lemmatizer.lemmatize(token.lower(), pos='n') for token in tokens]
        else:
            lemmatized = []
            for token, pos in pos_tags:
                # Map NLTK POS tags to WordNet POS tags
                wordnet_pos = self._get_wordnet_pos(pos)
                if wordnet_pos:
                    lemmatized.append(self.lemmatizer.lemmatize(token.lower(), pos=wordnet_pos))
                else:
                    lemmatized.append(self.lemmatizer.lemmatize(token.lower()))
            return lemmatized
    
    @staticmethod
    def _get_wordnet_pos(pos_tag: str) -> str:
        """
        Convert NLTK POS tags to WordNet POS tags
        
        Args:
            pos_tag: NLTK POS tag
            
        Returns:
            WordNet POS tag
        """
        from nltk.corpus import wordnet
        
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    
    def preprocess_complete(self, text: str, 
                           apply_stemming: bool = False,
                           apply_lemmatization: bool = True,
                           remove_stops: bool = True) -> List[str]:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text
            apply_stemming: Whether to apply stemming
            apply_lemmatization: Whether to apply lemmatization
            remove_stops: Whether to remove stopwords
            
        Returns:
            Preprocessed tokens
        """
        # Step 1: Normalize
        normalized = self.normalize_text(text)
        
        # Step 2: Tokenize
        tokens = self.tokenize(normalized)
        
        # Step 3: Remove stopwords
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        
        # Step 4: Stemming (optional)
        if apply_stemming:
            tokens = self.stem(tokens)
        
        # Step 5: Lemmatization (optional)
        if apply_lemmatization:
            pos_tags = pos_tag(tokens)
            tokens = self.lemmatize(tokens, pos_tags)
        
        return tokens
