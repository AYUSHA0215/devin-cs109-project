import numpy as np
from gensim.models import Word2Vec

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X_train, y_train, learning_rate=0.01, num_iterations=1000, lambda_param=0.01):
    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0

    for i in range(num_iterations):
        linear_model = np.dot(X_train, weights) + bias
        y_predicted = sigmoid(linear_model)

        dw = (1 / n_samples) * np.dot(X_train.T, (y_predicted - y_train)) + (lambda_param / n_samples) * weights
        db = (1 / n_samples) * np.sum(y_predicted - y_train)

        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias

def predict_logistic_regression(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    y_predicted = sigmoid(linear_model)
    y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
    return np.array(y_predicted_class)

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

def main():
    from preprocess_emails import load_and_preprocess_emails, transform_to_word2vec, train_test_split

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

    # Load the Word2Vec model
    word2vec_model = Word2Vec.load("word2vec.model")

    # Transform emails to Word2Vec vectors
    email_vectors = transform_to_word2vec(all_emails, word2vec_model)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(email_vectors, labels)

    # Print the number of spam and ham emails in the training and testing sets
    print(f"Training set: {np.sum(y_train == 0)} ham, {np.sum(y_train == 1)} spam")
    print(f"Testing set: {np.sum(y_test == 0)} ham, {np.sum(y_test == 1)} spam")

    # Train and evaluate Logistic Regression classifier
    weights, bias = train_logistic_regression(X_train, y_train)
    y_pred = predict_logistic_regression(X_test, weights, bias)
    accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
    print(f"Logistic Regression - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

if __name__ == "__main__":
    main()
