import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
except:
    pass

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b|\b\d{10}\b', '', text)
        
        # Remove punctuation and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords and stem
        words = [self.stemmer.stem(word) for word in text.split() 
                if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)

def feature_engineering(df):
    """Add additional features"""
    df = df.copy()
    
    # Text length features
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    # Spam indicator features
    df['has_urgent'] = df['text'].str.contains(r'\b(urgent|hurry|act now|limited time)\b', case=False, na=False)
    df['has_money'] = df['text'].str.contains(r'\b(money|cash|prize|win|free|\$)\b', case=False, na=False)
    df['has_caps'] = df['text'].str.contains(r'[A-Z]{3,}', na=False)
    df['exclamation_count'] = df['text'].str.count('!')
    
    return df

def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset"""
    print("📊 Loading dataset...")
    df = pd.read_csv(filepath, encoding='latin-1')[["v1", "v2"]]
    df.columns = ["label", "text"]
    df["label"] = df["label"].map({'ham': 0, 'spam': 1})
    
    # Remove duplicates and null values
    df = df.drop_duplicates()
    df = df.dropna()
    
    print(f"📈 Dataset shape: {df.shape}")
    print(f"📊 Spam ratio: {df['label'].mean():.2%}")
    
    # Preprocess text
    preprocessor = TextPreprocessor()
    df['text'] = df['text'].apply(preprocessor.preprocess_text)
    
    # Add features
    df = feature_engineering(df)
    
    return df

def train_multiple_models(X_train, X_test, y_train, y_test):
    """Train and compare multiple models"""
    models = {
        'Naive Bayes': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', MultinomialNB())
        ]),
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        'Random Forest': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }
    
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\n🤖 Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train and test
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"   Test Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = model
            best_model_name = name
    
    return best_model, best_model_name, best_score

def evaluate_model(model, X_test, y_test):
    """Detailed model evaluation"""
    predictions = model.predict(X_test)
    
    print("\n📊 Detailed Evaluation:")
    print(f"Accuracy: {model.score(X_test, y_test):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Ham', 'Spam']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    
    return predictions

def main():
    # Load and preprocess data
    df = load_and_preprocess_data("spam.csv")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    # Train multiple models
    best_model, best_model_name, best_score = train_multiple_models(
        X_train, X_test, y_train, y_test
    )
    
    print(f"\n🏆 Best Model: {best_model_name} with score: {best_score:.4f}")
    
    # Detailed evaluation
    evaluate_model(best_model, X_test, y_test)
    
    # Save the best model
    joblib.dump(best_model, "spam_model.joblib")
    print("\n💾 Model saved as 'spam_model.joblib'")
    
    # Save model metadata
    metadata = {
        'model_type': best_model_name,
        'accuracy': best_score,
        'training_size': len(X_train),
        'test_size': len(X_test)
    }
    joblib.dump(metadata, "model_metadata.joblib")
    print("📝 Model metadata saved")

if __name__ == "__main__":
    main()