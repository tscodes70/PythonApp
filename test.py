import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK resources if needed
nltk.download('punkt')
nltk.download('stopwords')

# Load the CSV file
df = pd.read_csv('data.csv')  # Replace 'your_data.csv' with the path to your CSV file

# Extract the text data from the "reviews.text" column
text_data = df['reviews.text']

# Convert the ratings into binary sentiment labels based on a threshold
threshold = 3.0  # Example threshold, adjust as needed
labels = ['positive' if rating >= threshold else 'negative' for rating in df['reviews.rating']]

# Preprocess the text data, handling missing values
stop_words = set(stopwords.words("english"))
filtered_words = []

for text in text_data:
    if isinstance(text, str):  # Check if the text is a string and not NaN
        tokens = word_tokenize(text.lower())
        filtered_words.append(' '.join([word for word in tokens if word.isalpha() and word not in stop_words]))
    else:
        filtered_words.append('')  # Replace missing values with an empty string

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=3000)

# Split data into training and testing sets
X = filtered_words
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TF-IDF representations of the text data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Naive Bayes classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = naive_bayes.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
