import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
import logging
import joblib
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Initialize NLTK components
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        self.use_stemming = self.config["preprocessing"]["use_stemming"]
        self.use_lemmatization = self.config["preprocessing"]["use_lemmatization"]
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.config["preprocessing"]["max_features"],
            stop_words='english'
        )

    def clean_text(self, text: str) -> str:
        """
        Lowercases, removes punctuation, special characters, and stopwords.
        """
        if not isinstance(text, str):
            return ""
            
        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and remove stopwords
        words = text.split()
        words = [w for w in words if w not in self.stop_words]
        
        # Stemming or Lemmatization
        if self.use_stemming:
            words = [self.stemmer.stem(w) for w in words]
        elif self.use_lemmatization:
            words = [self.lemmatizer.lemmatize(w) for w in words]
            
        return " ".join(words)

    def fit_transform(self, texts: List[str]):
        logger.info("Fitting and transforming texts with TF-IDF...")
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: List[str]):
        return self.vectorizer.transform(texts)

    def save_vectorizer(self, path: str):
        joblib.dump(self.vectorizer, path)
        logger.info(f"Vectorizer saved to {path}")

    def load_vectorizer(self, path: str):
        self.vectorizer = joblib.load(path)
        logger.info(f"Vectorizer loaded from {path}")

if __name__ == "__main__":
    # Simple test
    tp = TextPreprocessor()
    sample = "This is a SAMPLE news article! It's very fake, isn't it?"
    print(f"Original: {sample}")
    print(f"Cleaned: {tp.clean_text(sample)}")
