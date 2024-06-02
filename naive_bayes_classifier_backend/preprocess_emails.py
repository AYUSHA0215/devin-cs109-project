import os
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
import scipy.sparse as sp
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
import pickle

# Download NLTK data files (only the first time)
nltk.download('stopwords')
nltk.download('punkt')

# Initialize the stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Directories containing the email datasets
easy_ham_dir = 'easy_ham'
hard_ham_dir = 'hard_ham'
spam_dir = 'spam'

def upsample_minority_class(df_majority, df_minority):
    n_samples = len(df_majority)
    indices = np.random.choice(df_minority.index, size=n_samples, replace=True)
    df_minority_upsampled = df_minority.loc[indices]
    return df_minority_upsampled

def preprocess_email(text):
    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Remove stop words and perform stemming
    words = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]

    # Generate n-grams (bigrams and trigrams)
    bigrams = [' '.join(bigram) for bigram in nltk.bigrams(words)]
    trigrams = [' '.join(trigram) for trigram in nltk.trigrams(words)]

    # Combine unigrams, bigrams, and trigrams
    all_ngrams = words + bigrams + trigrams

    return ' '.join(all_ngrams)

def load_and_preprocess_emails(directory):
    emails = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', errors='ignore') as file:
            email_content = file.read()
            processed_content = preprocess_email(email_content)
            emails.append(processed_content)
    print(f"Loaded {len(emails)} emails from {directory}")
    if emails:
        print(f"Sample preprocessed email from {directory}: {emails[0]}")
    return emails

# Train a new Word2Vec model
def train_word2vec_model(emails, size=100, window=5, min_count=1, workers=4):
    sentences = [email.split() for email in emails]
    model = Word2Vec(sentences, vector_size=size, window=window, min_count=min_count, workers=workers)
    model.save('word2vec.model')
    return model

# Load and preprocess emails
easy_ham_emails = load_and_preprocess_emails(easy_ham_dir)
hard_ham_emails = load_and_preprocess_emails(hard_ham_dir)
spam_emails = load_and_preprocess_emails(spam_dir)

# Combine all emails for training the Word2Vec model
all_emails = easy_ham_emails + hard_ham_emails + spam_emails

# Train and save the Word2Vec model
word2vec_model = train_word2vec_model(all_emails)

def transform_to_word2vec(emails, model):
    vector_size = model.vector_size
    email_vectors = np.zeros((len(emails), vector_size))

    for i, email in enumerate(emails):
        words = email.split()
        bigrams = [' '.join(bigram) for bigram in nltk.bigrams(words)]
        trigrams = [' '.join(trigram) for trigram in nltk.trigrams(words)]
        all_ngrams = words + bigrams + trigrams
        word_vectors = [model.wv[word] for word in all_ngrams if word in model.wv]
        if word_vectors:
            email_vectors[i] = np.mean(word_vectors, axis=0)
        else:
            email_vectors[i] = np.zeros(vector_size)  # Ensure the vector has the correct size

    # Print the first 5 email vectors for inspection
    for i in range(min(5, len(email_vectors))):
        print(f"Email {i+1} vector: {email_vectors[i]}")

    return email_vectors

def calculate_tfidf(bow_matrix):
    tf = bow_matrix * (1 / bow_matrix.sum(axis=1, keepdims=True))
    df = np.array((bow_matrix > 0).sum(axis=0)).flatten()
    idf = np.log((1 + bow_matrix.shape[0]) / (1 + df)) + 1
    tfidf = tf * idf
    return tfidf

def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def train_naive_bayes(X_train, y_train, alpha=1.0):
    n_samples, n_features = X_train.shape
    n_classes = len(np.unique(y_train))
    priors = np.zeros(n_classes)
    likelihoods = np.zeros((n_classes, n_features))

    for c in range(n_classes):
        X_c = X_train[y_train == c]
        priors[c] = X_c.shape[0] / n_samples
        likelihoods[c, :] = (np.sum(X_c, axis=0) + alpha) / (np.sum(X_c) + alpha * n_features + 1e-9)  # Add a small constant to the denominator

        # Ensure all likelihoods are non-negative
        likelihoods[c, :] = np.maximum(likelihoods[c, :], 1e-9)

        # Log the values of X_c, priors, and likelihoods for debugging
        print(f"Class {c}:")
        print(f"X_c shape: {X_c.shape}")
        print(f"Priors[{c}]: {priors[c]}")
        print(f"Likelihoods[{c}, :]: {likelihoods[c, :]}")

    # Create and save the vocabulary
    vocabulary = {word: idx for idx, word in enumerate(np.unique(X_train))}
    with open('vocabulary.pkl', 'wb') as vocab_file:
        pickle.dump(vocabulary, vocab_file)

    # Log the size of the vocabulary and the shape of the likelihoods matrix
    print(f"Vocabulary size: {n_features}")
    print(f"Likelihoods shape: {likelihoods.shape}")
    print(f"Vocabulary: {vocabulary}")  # Added diagnostic print statement

    return priors, likelihoods

def predict_naive_bayes(X, priors, likelihoods):
    log_priors = np.log(priors)
    log_likelihoods = np.log(likelihoods)
    log_posteriors = X @ log_likelihoods.T + log_priors
    return np.argmax(log_posteriors, axis=1)

def calculate_metrics(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    if np.sum(y_pred == 1) == 0:
        precision = 0.0
    else:
        precision = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def predict_logistic_regression(model, X):
    return model.predict(X)

def main():
    # Directories containing the email datasets
    easy_ham_dir = 'easy_ham'
    hard_ham_dir = 'hard_ham'
    spam_dir = 'spam'

    # Load and preprocess emails
    easy_ham_emails = load_and_preprocess_emails(easy_ham_dir)
    hard_ham_emails = load_and_preprocess_emails(hard_ham_dir)
    spam_emails = load_and_preprocess_emails(spam_dir)

    # Combine all emails and create labels
    all_emails = easy_ham_emails + hard_ham_emails + spam_emails
    labels = np.array([0] * len(easy_ham_emails) + [0] * len(hard_ham_emails) + [1] * len(spam_emails))

    # Transform emails to Word2Vec vectors
    email_vectors = transform_to_word2vec(all_emails, word2vec_model)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(email_vectors, labels)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

    # Train and evaluate Naive Bayes classifier
    alphas = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for alpha in alphas:
        print(f"Training Naive Bayes with alpha={alpha}")
        priors, likelihoods = train_naive_bayes(X_train, y_train, alpha=alpha)
        with open('priors.pkl', 'wb') as f:
            pickle.dump(priors, f)
        with open('likelihoods.pkl', 'wb') as f:
            pickle.dump(likelihoods, f)
        y_pred = predict_naive_bayes(X_test, priors, likelihoods)
        accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
        print(f"Alpha: {alpha}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

        # Save the metrics to a file
        with open('metrics.pkl', 'wb') as metrics_file:
            pickle.dump({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}, metrics_file)
        print(f"Metrics saved to metrics.pkl: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    # Train and evaluate Logistic Regression classifier
    print("Training Logistic Regression")
    lr_model = train_logistic_regression(X_train, y_train)
    y_pred_lr = predict_logistic_regression(lr_model, X_test)
    accuracy_lr, precision_lr, recall_lr, f1_lr = calculate_metrics(y_test, y_pred_lr)
    print(f"Logistic Regression - Accuracy: {accuracy_lr}, Precision: {precision_lr}, Recall: {recall_lr}, F1 Score: {f1_lr}")

if __name__ == "__main__":
    main()
