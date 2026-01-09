"""
Demonstration of all preprocessing techniques

This script demonstrates all text preprocessing techniques on sample text.
"""

from text_preprocessing import TextPreprocessor
from nltk.tag import pos_tag


def demo_all_techniques():
    """Demonstrate all preprocessing techniques"""
    
    # Sample text
    sample_text = """
    Natural Language Processing (NLP) is a fascinating field of artificial intelligence 
    that deals with the interaction between computers and human language. NLP is used in 
    various applications such as machine translation, sentiment analysis, and question 
    answering systems. Processing textual data efficiently is crucial for building robust 
    NLP applications. Text normalization, tokenization, and lemmatization are fundamental 
    preprocessing steps that help improve model performance and reduce dimensionality.
    """
    
    preprocessor = TextPreprocessor()
    
    print("=" * 80)
    print("TEXT PREPROCESSING DEMONSTRATION")
    print("=" * 80)
    
    print("\n1. ORIGINAL TEXT:")
    print("-" * 80)
    print(sample_text.strip())
    
    # Text Normalization
    print("\n\n2. TEXT NORMALIZATION:")
    print("-" * 80)
    normalized = preprocessor.normalize_text(sample_text)
    print(normalized)
    print(f"Original length: {len(sample_text)} characters")
    print(f"Normalized length: {len(normalized)} characters")
    
    # Tokenization
    print("\n\n3. TOKENIZATION (Word Tokens):")
    print("-" * 80)
    tokens = preprocessor.tokenize(normalized)
    print(f"Total tokens: {len(tokens)}")
    print(f"First 20 tokens: {tokens[:20]}")
    
    # Sentence Tokenization
    print("\n\n4. SENTENCE TOKENIZATION:")
    print("-" * 80)
    sentences = preprocessor.tokenize(sample_text, use_sentence=True)
    print(f"Total sentences: {len(sentences)}")
    for i, sent in enumerate(sentences, 1):
        print(f"  Sentence {i}: {sent.strip()}")
    
    # Stopword Removal
    print("\n\n5. STOPWORD REMOVAL:")
    print("-" * 80)
    tokens_no_stops = preprocessor.remove_stopwords(tokens)
    print(f"Tokens before stopword removal: {len(tokens)}")
    print(f"Tokens after stopword removal: {len(tokens_no_stops)}")
    print(f"Stopwords removed: {len(tokens) - len(tokens_no_stops)}")
    print(f"Tokens after removal (first 20): {tokens_no_stops[:20]}")
    
    # Stemming
    print("\n\n6. STEMMING (Porter Stemmer):")
    print("-" * 80)
    stemmed = preprocessor.stem(tokens_no_stops)
    print(f"Sample stemmed tokens:")
    for original, stem in zip(tokens_no_stops[:15], stemmed[:15]):
        print(f"  {original:15} -> {stem}")
    print(f"Unique tokens before stemming: {len(set(tokens_no_stops))}")
    print(f"Unique tokens after stemming: {len(set(stemmed))}")
    
    # Lemmatization (without POS tags)
    print("\n\n7. LEMMATIZATION (without POS tags):")
    print("-" * 80)
    lemmatized = preprocessor.lemmatize(tokens_no_stops)
    print(f"Sample lemmatized tokens:")
    for original, lemma in zip(tokens_no_stops[:15], lemmatized[:15]):
        print(f"  {original:15} -> {lemma}")
    print(f"Unique tokens before lemmatization: {len(set(tokens_no_stops))}")
    print(f"Unique tokens after lemmatization: {len(set(lemmatized))}")
    
    # Lemmatization (with POS tags)
    print("\n\n8. LEMMATIZATION (with POS tags):")
    print("-" * 80)
    pos_tags = pos_tag(tokens_no_stops)
    lemmatized_with_pos = preprocessor.lemmatize(tokens_no_stops, pos_tags)
    print(f"Sample tokens with POS tags and lemmatization:")
    for (token, pos), lemma in zip(pos_tags[:15], lemmatized_with_pos[:15]):
        print(f"  {token:15} ({pos:3}) -> {lemma}")
    print(f"Unique tokens with POS lemmatization: {len(set(lemmatized_with_pos))}")
    
    # Complete Pipeline
    print("\n\n9. COMPLETE PREPROCESSING PIPELINE:")
    print("-" * 80)
    complete = preprocessor.preprocess_complete(sample_text, 
                                               apply_stemming=False,
                                               apply_lemmatization=True,
                                               remove_stops=True)
    print(f"Final tokens: {complete}")
    print(f"Total final tokens: {len(complete)}")
    print(f"Unique final tokens: {len(set(complete))}")
    
    # Comparison: Stemming vs Lemmatization
    print("\n\n10. STEMMING VS LEMMATIZATION COMPARISON:")
    print("-" * 80)
    test_words = ['running', 'runs', 'ran', 'processing', 'processed', 'studies', 'studying']
    print(f"{'Original':15} {'Stemmed':15} {'Lemmatized':15}")
    print("-" * 45)
    for word in test_words:
        stemmed_word = preprocessor.stemmer.stem(word)
        lemmatized_word = preprocessor.lemmatizer.lemmatize(word)
        print(f"{word:15} {stemmed_word:15} {lemmatized_word:15}")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    demo_all_techniques()
