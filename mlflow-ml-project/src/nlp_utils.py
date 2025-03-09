def tokenize_text(text):
    """Tokenizes the input text into words."""
    import re
    from nltk.tokenize import word_tokenize
    # Normalize the text
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    return tokens

def vectorize_texts(texts, vectorizer=None):
    """Vectorizes a list of texts using the provided vectorizer or creates a new one."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        vectorized_texts = vectorizer.fit_transform(texts)
    else:
        vectorized_texts = vectorizer.transform(texts)
    return vectorized_texts, vectorizer

def extract_features(dataframe):
    """Extracts features from the dataframe for NLP tasks."""
    # Assuming the dataframe has a column 'abstract' for feature extraction
    abstracts = dataframe['abstract'].tolist()
    vectorized_abstracts, vectorizer = vectorize_texts(abstracts)
    return vectorized_abstracts, vectorizer

def preprocess_texts(texts):
    """Preprocesses a list of texts for NLP tasks."""
    return [tokenize_text(text) for text in texts]