import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
from gensim.models import Word2Vec
import pickle
from preprocess_emails import load_and_preprocess_emails, transform_to_word2vec, train_test_split, calculate_tfidf, calculate_metrics
from logistic_regression_classifier import train_logistic_regression, predict_logistic_regression, sigmoid

# Initialize the Flask app and CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Set up logging
logging.basicConfig(filename='flask_app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

@app.before_request
def handle_options_request():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        headers = None
        if 'ACCESS_CONTROL_REQUEST_HEADERS' in request.headers:
            headers = request.headers['ACCESS_CONTROL_REQUEST_HEADERS']
        response.headers['Access-Control-Allow-Headers'] = headers
        response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS, DELETE'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = "https://eloquent-dolphin-906f54.netlify.app"
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS, DELETE'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    # Log the response headers to confirm CORS headers are being added
    logging.debug(f"Response headers: {response.headers}")
    # Log the request URL to confirm the Flask application's URL
    logging.debug(f"Request URL: {request.url}")
    return response

# Ensure CORS headers are added to all responses, including error responses
@app.errorhandler(Exception)
def handle_exception(e):
    response = jsonify({'error': str(e)})
    return add_cors_headers(response)

# Initialize the stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocess email function
def preprocess_email(text):
    # Log the type of the input text
    print(f"Type of input text in preprocess_email: {type(text)}")

    # Ensure the input text is a string
    if isinstance(text, np.ndarray):
        text = ' '.join(map(str, text))  # Join array items into a single string
    text = str(text)  # Explicitly convert to string

    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    words = nltk.word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

# Create Bag of Words model
def create_bag_of_words(emails):
    vocabulary = {}
    for email in emails:
        for word in email.split():
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)
    return vocabulary

# Transform emails to Bag of Words matrix
def transform_to_bow(emails, vocabulary):
    bow_matrix = np.zeros((len(emails), len(vocabulary)))
    for i, email in enumerate(emails):
        for word in email.split():
            if word in vocabulary:
                bow_matrix[i, vocabulary[word]] += 1
    # Ensure the BoW matrix has the correct number of features
    assert bow_matrix.shape[1] == len(vocabulary), f"BoW matrix shape {bow_matrix.shape} does not match vocabulary size {len(vocabulary)}"
    return bow_matrix

# Calculate TF-IDF values
def calculate_tfidf(bow_matrix):
    row_sums = np.sum(bow_matrix, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10  # Replace zero sums with a small constant to avoid division by zero
    tf = bow_matrix / row_sums
    df = np.sum(bow_matrix > 0, axis=0)
    idf = np.log((1 + len(bow_matrix)) / (1 + df)) + 1
    tfidf = tf * idf
    tfidf[np.isnan(tfidf)] = 0  # Replace NaN values with zeros
    tfidf[np.isinf(tfidf)] = 0  # Replace inf values with zeros
    # Ensure the TF-IDF matrix has the correct number of features
    assert tfidf.shape[1] == bow_matrix.shape[1], f"TF-IDF matrix shape {tfidf.shape} does not match BoW matrix shape {bow_matrix.shape}"
    return tfidf

# Train Naive Bayes classifier
def train_naive_bayes(X_train, y_train):
    n_samples, n_features = X_train.shape
    n_classes = len(np.unique(y_train))
    priors = np.zeros(n_classes)
    likelihoods = np.zeros((n_classes, n_features))

    for c in range(n_classes):
        X_c = X_train[y_train == c]
        priors[c] = X_c.shape[0] / n_samples
        likelihoods[c, :] = (np.sum(X_c, axis=0) + 1) / (np.sum(X_c) + n_features)

    return priors, likelihoods

# Predict using Naive Bayes classifier
def predict_naive_bayes(X, priors, likelihoods):
    log_priors = np.log(priors)
    log_likelihoods = np.log(likelihoods)
    print(f"email_tfidf shape: {X.shape}")
    print(f"priors shape: {priors.shape}")
    print(f"likelihoods shape: {likelihoods.shape}")
    print(f"email_tfidf: {X}")
    print(f"priors: {priors}")
    print(f"likelihoods: {likelihoods}")

    # Ensure the input vector and likelihoods have compatible shapes
    if X.shape[1] != likelihoods.shape[1]:
        raise ValueError(f"Shape misalignment: email_tfidf shape {X.shape} and likelihoods shape {likelihoods.shape} are not aligned")

    log_posteriors = X @ log_likelihoods.T + log_priors
    return np.argmax(log_posteriors, axis=1)

# Load and preprocess emails from a directory
def load_and_preprocess_emails(directory):
    emails = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='latin-1') as file:
            content = file.read()
            print(f"Type of content read from file {filename}: {type(content)}")
            emails.append(preprocess_email(content))
    return emails

# Load the preprocessed emails and train the model
easy_ham_dir = '/home/ubuntu/easy_ham'
hard_ham_dir = '/home/ubuntu/hard_ham'
spam_dir = '/home/ubuntu/spam'

easy_ham_emails = load_and_preprocess_emails(easy_ham_dir)
hard_ham_emails = load_and_preprocess_emails(hard_ham_dir)
spam_emails = load_and_preprocess_emails(spam_dir)

all_emails = easy_ham_emails + hard_ham_emails + spam_emails
labels = np.array([0] * len(easy_ham_emails) + [0] * len(hard_ham_emails) + [1] * len(spam_emails))

df = pd.DataFrame({'email': all_emails, 'label': labels})

df_majority = df[df.label == 0]
df_minority = df[df.label == 1]

def upsample_minority_class(df_majority, df_minority):
    n_samples = len(df_majority)
    indices = np.random.choice(df_minority.index, size=n_samples, replace=True)
    df_minority_upsampled = df_minority.loc[indices]
    return df_minority_upsampled

df_minority_upsampled = upsample_minority_class(df_majority, df_minority)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

vocabulary = create_bag_of_words(df_upsampled['email'])
X_counts = transform_to_bow(df_upsampled['email'], vocabulary)
X_tfidf = calculate_tfidf(X_counts)

# Train Naive Bayes classifier and calculate metrics
priors, likelihoods = train_naive_bayes(X_tfidf, df_upsampled['label'].values)
y_pred = predict_naive_bayes(X_tfidf, priors, likelihoods)
accuracy, precision, recall, f1_score = calculate_metrics(df_upsampled['label'].values, y_pred)

# Save the vocabulary, model parameters, and metrics after training
with open('vocabulary.pkl', 'wb') as vocab_file:
    pickle.dump(vocabulary, vocab_file)

with open('priors.npy', 'wb') as priors_file:
    np.save(priors_file, priors)

with open('likelihoods.npy', 'wb') as likelihoods_file:
    np.save(likelihoods_file, likelihoods)

with open('metrics.pkl', 'wb') as metrics_file:
    pickle.dump({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score}, metrics_file)

# Load the vocabulary and model parameters from files at the start of the script
with open('vocabulary.pkl', 'rb') as vocab_file:
    vocabulary = pickle.load(vocab_file)

priors = np.load('priors.npy')
likelihoods = np.load('likelihoods.npy')

with open('metrics.pkl', 'rb') as metrics_file:
    metrics = pickle.load(metrics_file)
accuracy = metrics['accuracy']
precision = metrics['precision']
recall = metrics['recall']
f1_score = metrics['f1_score']

# Print the sizes of the loaded vocabulary and the likelihoods matrix
print(f"Loaded vocabulary size: {len(vocabulary)}")
print(f"Likelihoods shape: {likelihoods.shape}")

# Flask route to classify email
@app.route('/classify_naive_bayes', methods=['POST'])
def classify_naive_bayes():
    try:
        print("classify_naive_bayes function triggered")
        data = request.get_json()
        print(f"Incoming request data: {data}")
        email_text = data.get('email_text', '')
        print(f"Type of email_text immediately after extraction: {type(email_text)}")
        print(f"Content of email_text immediately after extraction: {email_text}")

        # Ensure email_text is a string immediately after extraction
        if isinstance(email_text, np.ndarray):
            email_text = ' '.join(str(item) for item in email_text)  # Convert each element to string and join
        elif isinstance(email_text, (list, tuple)):
            email_text = ' '.join(map(str, email_text))  # Join list or tuple items into a single string
        email_text = str(email_text)  # Explicitly convert to string

        # Log the type of email_text after conversion
        print(f"Type of email_text after conversion: {type(email_text)}")
        print(f"Content of email_text after conversion: {email_text}")

        # Log the size of the loaded vocabulary and the shape of the likelihoods matrix
        print(f"Loaded vocabulary size: {len(vocabulary)}")
        print(f"Vocabulary: {vocabulary}")
        print(f"Likelihoods shape: {likelihoods.shape}")

        # Log the type and content of email_text before preprocessing
        print(f"Type of email_text before preprocessing: {type(email_text)}")
        print(f"Content of email_text before preprocessing: {email_text}")

        # Preprocess the email text
        preprocessed_email = preprocess_email(email_text)
        print(f"Preprocessed email: {preprocessed_email}")

        # Print the size of the vocabulary before transforming to BoW
        print(f"Vocabulary size before transform_to_bow: {len(vocabulary)}")

        # Transform the preprocessed email to BoW using the full vocabulary
        email_bow = transform_to_bow([preprocessed_email], vocabulary)
        print(f"email_bow shape: {email_bow.shape}")
        print(f"email_bow: {email_bow}")

        # Log the type and content of email_bow before TF-IDF transformation
        print(f"Type of email_bow before TF-IDF transformation: {type(email_bow)}")
        print(f"Content of email_bow before TF-IDF transformation: {email_bow}")

        email_tfidf = calculate_tfidf(email_bow)
        print(f"email_tfidf shape: {email_tfidf.shape}")
        print(f"email_tfidf: {email_tfidf}")

        # Ensure the input vector has the correct shape
        if email_tfidf.shape[1] != likelihoods.shape[1]:
            raise ValueError(f"Shape misalignment: email_tfidf shape {email_tfidf.shape} and likelihoods shape {likelihoods.shape} are not aligned")

        # Diagnostic print statements
        print(f"email_tfidf shape: {email_tfidf.shape}")
        print(f"priors shape: {priors.shape}")
        print(f"likelihoods shape: {likelihoods.shape}")
        print(f"email_tfidf: {email_tfidf}")
        print(f"priors: {priors}")
        print(f"likelihoods: {likelihoods}")
        print(f"Vocabulary: {vocabulary}")

        classification = predict_naive_bayes(email_tfidf, priors, likelihoods)[0]

        classification_label = 'spam' if classification == 1 else 'ham'
        print(f"Classification result: {classification_label}")  # Logging statement

        response = jsonify({
            'classification': classification_label,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        })
        print(f"Response: {response.get_json()}")  # Logging statement

        # Add diagnostic print statement to confirm CORS headers
        print(f"Response headers before returning: {response.headers}")

        return response
    except Exception as e:
        print(f"Error during classification: {e}")
        response = jsonify({'error': str(e)})
        print(f"Error response: {response.get_json()}")  # Logging statement

        # Add diagnostic print statement to confirm CORS headers
        print(f"Error response headers before returning: {response.headers}")

        return response, 500

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    try:
        data = request.get_json()
        email_text = data.get('email_text', '')
        feedback = data.get('feedback', '')

        if feedback == 'not spam':
            # Add the email to the ham dataset and retrain the model
            preprocessed_email = preprocess_email(email_text)
            df.loc[len(df)] = [preprocessed_email, 0]
            df_majority = df[df.label == 0]
            df_minority = df[df.label == 1]
            df_minority_upsampled = upsample_minority_class(df_majority, df_minority)
            df_upsampled = pd.concat([df_majority, df_minority_upsampled])
            vocabulary = create_bag_of_words(df_upsampled['email'])
            X_counts = transform_to_bow(df_upsampled['email'], vocabulary)
            X_tfidf = calculate_tfidf(X_counts)
            global priors, likelihoods
            priors, likelihoods = train_naive_bayes(X_tfidf, df_upsampled['label'].values)

            # Save the updated vocabulary and model parameters
            with open('vocabulary.pkl', 'wb') as vocab_file:
                pickle.dump(vocabulary, vocab_file)

            with open('priors.npy', 'wb') as priors_file:
                np.save(priors_file, priors)

            with open('likelihoods.npy', 'wb') as likelihoods_file:
                np.save(likelihoods_file, likelihoods)

        return jsonify({'message': 'Feedback submitted successfully.'})
    except Exception as e:
        print(f"Error during feedback handling: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/classify_logistic', methods=['POST'])
def classify_email_logistic():
    try:
        data = request.get_json()
        print(f"Incoming request data: {data}")
        email_text = data.get('email_text', '')

        # Load the Logistic Regression model parameters from files
        weights_path = os.path.join(os.path.dirname(__file__), 'logistic_regression_weights.npy')
        bias_path = os.path.join(os.path.dirname(__file__), 'logistic_regression_bias.npy')

        with open(weights_path, 'rb') as weights_file:
            weights = np.load(weights_file)
        with open(bias_path, 'rb') as bias_file:
            bias = np.load(bias_file)

        # Load the Word2Vec model
        word2vec_model = Word2Vec.load("word2vec.model")

        preprocessed_email = preprocess_email(email_text)
        email_vector = transform_to_word2vec([preprocessed_email], word2vec_model)

        # Reshape email_vector and weights to ensure compatibility for matrix multiplication
        email_vector = email_vector.reshape(1, -1)
        weights = weights.reshape(-1, 1)

        # Diagnostic print statements
        print(f"email_vector shape: {email_vector.shape}")
        print(f"weights shape: {weights.shape}")
        print(f"bias shape: {bias.shape}")

        # Ensure the shapes are compatible for matrix multiplication
        if email_vector.shape[1] != weights.shape[0]:
            raise ValueError(f"Shape misalignment: email_vector shape {email_vector.shape} and weights shape {weights.shape} are not aligned")

        classification = predict_logistic_regression(email_vector, weights, bias)[0]

        classification_label = 'spam' if classification == 1 else 'ham'
        print(f"Classification result: {classification_label}")  # Logging statement

        # Load the test dataset and true labels
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')

        # Make predictions on the test dataset
        y_pred = predict_logistic_regression(X_test, weights, bias)

        # Calculate evaluation metrics
        accuracy = np.mean(y_test == y_pred)
        precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1)
        recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)
        f1_score = 2 * (precision * recall) / (precision + recall)

        response = jsonify({
            'classification': classification_label,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        })
        print(f"Response: {response.get_json()}")  # Logging statement
        return response
    except Exception as e:
        print(f"Error during classification: {e}")
        response = jsonify({'error': str(e)})
        print(f"Error response: {response.get_json()}")  # Logging statement
        return response, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
