# NLP Text Preprocessing Implementation

A comprehensive implementation of text preprocessing techniques for multiple NLP datasets.

## Overview

This project implements essential NLP preprocessing techniques:
- **Text Normalization** - Lowercase conversion, special character removal, URL/HTML removal
- **Tokenization** - Word and sentence-level tokenization
- **Stopword Removal** - Filtering common English words
- **Stemming** - Porter Stemmer algorithm for word reduction
- **Lemmatization** - WordNet Lemmatizer for morphological analysis

## Datasets Supported

### 1. Brown Corpus
- **Description**: A balanced corpus of English with ~1 million words across 15 categories
- **Module**: `brown_corpus_preprocessing.py`
- **Features**: Category-based processing, detailed statistics per category

### 2. Universal Dependencies English Treebank
- **Description**: Syntactically-annotated English corpus following UD guidelines
- **Module**: `ud_english_treebank_preprocessing.py`
- **Features**: Dependency tree parsing, sentence-level analysis

### 3. 20 Newsgroups Dataset
- **Description**: ~18,000 newsgroup posts across 20 categories
- **Module**: `newsgroup_preprocessing.py`
- **Features**: Category filtering, document-level processing, header/footer removal

### 4. Reuters Dataset
- **Description**: Large collection of news documents for text categorization
- **Module**: `reuters_preprocessing.py`
- **Features**: Multi-category support, comprehensive statistics

## Project Structure

```
NLP/
├── requirements.txt                          # Python dependencies
├── text_preprocessing.py                     # Core preprocessing module
├── brown_corpus_preprocessing.py             # Brown Corpus processor
├── ud_english_treebank_preprocessing.py      # UD Treebank processor
├── newsgroup_preprocessing.py                # 20 Newsgroups processor
├── reuters_preprocessing.py                  # Reuters processor
├── demo_preprocessing.py                     # Interactive demonstration
├── run_all_preprocessing.py                  # Master execution script
├── reports/                                  # Output reports directory
│   ├── brown_corpus_report.csv
│   ├── ud_english_treebank_stats.csv
│   ├── newsgroups_category_stats.csv
│   ├── reuters_category_stats.csv
│   └── datasets_comparison.csv
└── README.md                                 # This file
```

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **NLTK data will be automatically downloaded** on first run

## Usage

### Option 1: Run All Datasets (Recommended)
```bash
python run_all_preprocessing.py
```

### Option 2: Interactive Demonstration
```bash
python demo_preprocessing.py
```

This shows all preprocessing steps on sample text.

### Option 3: Process Individual Datasets

#### Brown Corpus
```python
from brown_corpus_preprocessing import BrownCorpusProcessor

processor = BrownCorpusProcessor()
report = processor.generate_preprocessing_report()
print(report)
```

#### Universal Dependencies English Treebank
```python
from ud_english_treebank_preprocessing import UDEnglishTreebankProcessor

processor = UDEnglishTreebankProcessor()
summary = processor.get_preprocessing_summary()
print(summary)
```

#### 20 Newsgroups
```python
from newsgroup_preprocessing import NewsgroupProcessor

processor = NewsgroupProcessor()
processor.download_dataset()
stats = processor.get_category_statistics()
print(stats)
```

#### Reuters
```python
from reuters_preprocessing import ReutersProcessor

processor = ReutersProcessor()
summary = processor.get_preprocessing_summary()
print(summary)
```

### Option 4: Core Preprocessing Module
```python
from text_preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()

# Text Normalization
normalized = preprocessor.normalize_text("Your text here...")

# Tokenization
tokens = preprocessor.tokenize(normalized)

# Stopword Removal
filtered = preprocessor.remove_stopwords(tokens)

# Stemming
stemmed = preprocessor.stem(filtered)

# Lemmatization
lemmatized = preprocessor.lemmatize(filtered)

# Complete Pipeline
result = preprocessor.preprocess_complete(
    "Your text here...",
    apply_stemming=False,
    apply_lemmatization=True,
    remove_stops=True
)
```

## Key Classes and Methods

### TextPreprocessor

**Methods**:
- `normalize_text(text)` - Normalize input text
- `tokenize(text, use_sentence)` - Tokenize into words or sentences
- `remove_stopwords(tokens)` - Filter stopwords
- `stem(tokens)` - Apply Porter Stemming
- `lemmatize(tokens, pos_tags)` - Apply WordNet Lemmatization
- `preprocess_complete(text, **kwargs)` - Complete pipeline

### Dataset Processors

Each processor follows this pattern:

**Common Methods**:
- `get_corpus_text()` - Get full corpus text
- `preprocess_corpus()` - Preprocess entire dataset
- `get_preprocessing_summary()` - Get overall statistics
- `get_category_statistics()` / `preprocess_by_category()` - Category-level analysis

## Preprocessing Pipeline Details

### 1. Text Normalization
- Converts text to lowercase
- Removes URLs and email addresses
- Removes HTML tags
- Removes special characters (keeps alphanumeric and spaces)
- Removes extra whitespace

### 2. Tokenization
- **Word Tokenization**: Splits text into individual tokens
- **Sentence Tokenization**: Splits text into sentences

### 3. Stopword Removal
- Removes common English words (the, is, at, etc.)
- Uses NLTK's English stopwords list
- Significantly reduces token count without losing meaning

### 4. Stemming
- Applies Porter Stemmer algorithm
- Removes word suffixes (e.g., "running" → "run")
- Fast and deterministic
- May produce non-words

### 5. Lemmatization
- Uses WordNet Lemmatizer
- Finds base form of words (e.g., "better" → "good")
- Can use POS tags for better accuracy
- Produces valid dictionary words

## Output Examples

### Brown Corpus Report
```
category  token_count  unique_tokens  unique_stemmed  unique_lemmatized  stopword_reduction_percent
news      100000       8500          5200            6400               34.5
```

### Dataset Comparison
```
Dataset                          Total Documents  Tokens      Unique Tokens  Avg Stopword Reduction %
Brown Corpus                     15               1000000     50000          35.2
UD English Treebank             2500             150000      12000          33.8
20 Newsgroups                   18000            5000000     75000          36.1
Reuters                         10788            2000000     40000          34.9
```

## Performance Characteristics

| Technique | Speed | Dimensionality Reduction | Information Loss |
|-----------|-------|--------------------------|------------------|
| Normalization | ⚡ Very Fast | - | Minimal |
| Tokenization | ⚡ Very Fast | - | None |
| Stopword Removal | ⚡ Very Fast | ~30-40% | Low |
| Stemming | ⚡ Fast | ~40-60% | Medium |
| Lemmatization | ⚠ Moderate | ~35-55% | Low |

## Tips for Best Results

1. **Always normalize first** - Ensures consistent input
2. **Use lemmatization for meaning preservation** - Better for semantic tasks
3. **Use stemming for efficiency** - Faster when dimensionality reduction is priority
4. **Consider context** - Different tasks may need different preprocessing
5. **Keep stopwords if relevant** - Domain-specific terms might be in stopword lists
6. **Use POS tags with lemmatization** - Improves accuracy significantly

## Dependencies

- **nltk**: Natural Language Toolkit for NLP tasks
- **scikit-learn**: ML library for dataset loading
- **spacy**: NLP library (optional for advanced usage)
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **regex**: Advanced regex operations

## Example Workflow

```python
from text_preprocessing import TextPreprocessor
from brown_corpus_preprocessing import BrownCorpusProcessor

# Initialize
preprocessor = TextPreprocessor()
brown_proc = BrownCorpusProcessor()

# Get sample text
text = brown_proc.get_category_text('news')

# Preprocess
cleaned_tokens = preprocessor.preprocess_complete(
    text,
    apply_stemming=False,
    apply_lemmatization=True,
    remove_stops=True
)

print(f"Processed {len(cleaned_tokens)} tokens")
print(f"Unique tokens: {len(set(cleaned_tokens))}")
```

## Advanced Usage

### Custom Stemming/Lemmatization
```python
from nltk.stem.snowball import SnowballStemmer

# Use Snowball stemmer instead
stemmer = SnowballStemmer('english')
stemmed = [stemmer.stem(token) for token in tokens]
```

### Processing with POS Tags
```python
from nltk.tag import pos_tag

tokens = preprocessor.tokenize(text)
pos_tags = pos_tag(tokens)
lemmatized = preprocessor.lemmatize(tokens, pos_tags)
```

## Troubleshooting

**Issue**: NLTK data not found
**Solution**: Data is automatically downloaded on first run. If issues persist:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

**Issue**: Memory error with large datasets
**Solution**: Process datasets in batches using category-level processing

**Issue**: Lemmatization produces unexpected results
**Solution**: Use POS tags for better accuracy

## Contributing

To add support for new datasets:

1. Create a new processor class inheriting from a base pattern
2. Implement required methods: `get_corpus_text()`, `preprocess_corpus()`
3. Add statistics generation methods
4. Update `run_all_preprocessing.py`

## License

MIT License

## Author

NLP Preprocessing Project

## References

- NLTK Documentation: https://www.nltk.org/
- Brown Corpus: https://en.wikipedia.org/wiki/Brown_Corpus
- Universal Dependencies: https://universaldependencies.org/
- 20 Newsgroups: http://qwone.com/~jason/20Newsgroups/
- Reuters Dataset: https://www.nltk.org/howto/reuters.html
