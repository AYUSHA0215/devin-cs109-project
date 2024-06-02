import os
import numpy as np
from gensim.models import Word2Vec
from preprocess_emails import load_and_preprocess_emails, transform_to_word2vec, train_test_split, train_logistic_regression

# Directories containing the email datasets
easy_ham_dir = 'easy_ham'
hard_ham_dir = 'hard_ham'
spam_dir = 'spam'

def main():
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

    # Train Logistic Regression classifier
    print("Training Logistic Regression")
    lr_model = train_logistic_regression(X_train, y_train)

    # Save the Logistic Regression model parameters
    np.save('logistic_regression_weights.npy', lr_model.coef_)
    np.save('logistic_regression_bias.npy', lr_model.intercept_)

    # Evaluate Logistic Regression classifier
    y_pred_lr = lr_model.predict(X_test)
    accuracy_lr = np.mean(y_test == y_pred_lr)
    precision_lr = np.sum((y_pred_lr == 1) & (y_test == 1)) / np.sum(y_pred_lr == 1)
    recall_lr = np.sum((y_pred_lr == 1) & (y_test == 1)) / np.sum(y_test == 1)
    f1_lr = 2 * (precision_lr * recall_lr) / (precision_lr + recall_lr)
    print(f"Logistic Regression - Accuracy: {accuracy_lr}, Precision: {precision_lr}, Recall: {recall_lr}, F1 Score: {f1_lr}")

if __name__ == "__main__":
    main()
