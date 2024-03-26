from flask import Flask, render_template, request
import pandas as pd
import re
from rake_nltk import Rake
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset from CSV file
dataset = pd.read_csv("good.csv")

# Convert text columns to lowercase
dataset = dataset.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# Remove non-word characters and digits from the 'text' column
dataset['preprocessed'] = dataset['text'].replace(to_replace=r'[^\w\s]', value='', regex=True)
dataset['preprocessed'] = dataset['preprocessed'].replace(to_replace=r'\d', value='', regex=True)

# Tokenize the preprocessed text
dataset['preprocessed'] = dataset['preprocessed'].apply(word_tokenize)

# Remove stopwords from the tokenized text
stop_words = set(stopwords.words('english'))
dataset['preprocessed'] = dataset['preprocessed'].apply(lambda x: [word for word in x if word not in stop_words])

# Initialize the Porter Stemmer for stemming
stemmer = PorterStemmer()

# Apply stemming to the tokenized text
dataset['preprocessed'] = dataset['preprocessed'].apply(lambda x: [stemmer.stem(word) for word in x])

# Initialize RAKE (Rapid Automatic Keyword Extraction)
r = Rake()

# Function to extract keywords using RAKE
def extract_keywords(tokens):
    text = ' '.join(tokens)  # Convert list of tokens back to string
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()[:5]  # Get top 5 keywords

# Apply the function to the 'preprocessed' column to extract keywords
dataset['keywords'] = dataset['preprocessed'].apply(extract_keywords)

# Convert the extracted keywords to string format for vectorization
dataset['keywords_str'] = dataset['keywords'].apply(lambda x: ' '.join(x))

# Vectorize the keywords using TF-IDF (Term Frequency-Inverse Document Frequency)
vectorizer = TfidfVectorizer(max_df=0.8, min_df=2)
X = vectorizer.fit_transform(dataset['keywords_str'])

# Apply K-Means clustering to categorize the articles into 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
dataset['category'] = kmeans.fit_predict(X) + 1  # Assign categories (1-5) to the articles

# Function to search for articles based on a query
def search_articles(query, data, tfidf_vectorizer, tfidf_matrix):
    # Preprocess the query
    query = query.lower()
    query_tokens = nltk.word_tokenize(query)
    query_tokens = [stemmer.stem(word) for word in query_tokens if word.isalnum() and word not in stop_words]
    query_text = ' '.join(query_tokens)

    # Convert the query to a TF-IDF vector
    query_vector = tfidf_vectorizer.transform([query_text])

    # Calculate the similarity between the query vector and article vectors using dot product
    similarities = tfidf_matrix.dot(query_vector.T).toarray().flatten()

    # Sort articles by similarity and get the top results
    top_indices = similarities.argsort()[::-1][:5]  # Get top 5 most similar articles
    top_articles = data.iloc[top_indices]

    return top_articles

# Flask route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    count = 0
    categories = sorted(dataset['category'].unique())
    
    if request.method == 'POST':
        query = request.form.get('query')
        results = search_articles(query, dataset, vectorizer, X)
        count = len(results)
    
    return render_template('index.html', categories=categories, results=results, count=count)

# Flask route for displaying articles by category
@app.route('/category/<int:category_id>')
def category(category_id):
    category_results = dataset[dataset['category'] == category_id]
    return render_template('category.html', category_results=category_results.to_dict('records'))

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
