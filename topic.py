import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load your dataset from a CSV file (assuming you have 'bbc-text.csv' with 'text' and 'category' columns)
dataset = pd.read_csv('bbc-text.csv', encoding='ISO-8859-1')

# Preprocess the text data
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

# Apply preprocessing to the entire dataset
dataset['preprocessed_text'] = dataset['text'].apply(preprocess_text)

# Split the data into training and testing sets (80% training, 20% testing)
train_data, test_data, train_labels, test_labels = train_test_split(
    dataset['preprocessed_text'], dataset['category'], test_size=0.2, random_state=42
)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
train_vectors = tfidf_vectorizer.fit_transform(train_data)
test_vectors = tfidf_vectorizer.transform(test_data)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(train_vectors, train_labels)

# Test the classifier
predictions = classifier.predict(test_vectors)
accuracy = metrics.accuracy_score(test_labels, predictions)
print(f"Classifier Accuracy: {accuracy}")

# Function to classify text using the trained classifier
def classify_text(input_text):
    preprocessed_input = preprocess_text(input_text)
    input_vector = tfidf_vectorizer.transform([preprocessed_input])
    topic_label = classifier.predict(input_vector)[0]
    return topic_label
