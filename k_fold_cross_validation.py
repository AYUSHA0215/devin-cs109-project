import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from preprocess_emails import (
    load_and_preprocess_emails,
    transform_to_word2vec,
    calculate_tfidf,
    calculate_metrics,
    upsample_minority_class
)
from svm_classifier import SVMClassifier
from sklearn.feature_selection import SelectKBest, chi2
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler

def k_fold_cross_validation(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Scale features to a non-negative range
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Feature selection to reduce the number of features
        selector = SelectKBest(chi2, k=1000)  # Select top 1000 features
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

        # Train the SVM classifier
        svm = SVMClassifier()
        svm.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = svm.predict(X_test)

        # Calculate evaluation metrics
        accuracy, precision, recall, f1 = calculate_metrics(y_test, predictions)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Print the average evaluation metrics
    print(f'Average Accuracy: {np.mean(accuracies)}')
    print(f'Average Precision: {np.mean(precisions)}')
    print(f'Average Recall: {np.mean(recalls)}')
    print(f'Average F1 Score: {np.mean(f1_scores)}')

def main():
    # Directories containing the email datasets
    easy_ham_dir = 'easy_ham'
    hard_ham_dir = 'hard_ham'
    spam_dir = 'spam'

    # Load and preprocess emails
    easy_ham_emails = load_and_preprocess_emails(easy_ham_dir)
    hard_ham_emails = load_and_preprocess_emails(hard_ham_dir)
    spam_emails = load_and_preprocess_emails(spam_dir)

    # Print the number of preprocessed emails in each category
    print(f'Easy Ham Emails: {len(easy_ham_emails)}')
    print(f'Hard Ham Emails: {len(hard_ham_emails)}')
    print(f'Spam Emails: {len(spam_emails)}')

    # Combine all emails and create labels
    all_emails = easy_ham_emails + hard_ham_emails + spam_emails
    labels = np.array([0] * len(easy_ham_emails) + [0] * len(hard_ham_emails) + [1] * len(spam_emails))

    # Create a DataFrame for the emails and labels
    df = pd.DataFrame({'email': all_emails, 'label': labels})

    # Print the class distribution
    print(f'Class distribution: {df["label"].value_counts()}')

    # Upsample the minority class (spam emails)
    df_majority = df[df.label == 0]
    df_minority = df[df.label == 1]
    df_minority_upsampled = upsample_minority_class(df_majority, df_minority)
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    # Load the Word2Vec model
    word2vec_model = Word2Vec.load('word2vec.model')

    # Transform emails to Word2Vec vectors
    X_word2vec = transform_to_word2vec(df_upsampled['email'], word2vec_model)

    # Apply TF-IDF transformation
    X_tfidf = calculate_tfidf(X_word2vec)

    # Print the shape of the feature matrix
    print(f'Feature matrix shape: {X_tfidf.shape}')

    # Perform k-fold cross-validation
    k_fold_cross_validation(X_tfidf, df_upsampled['label'].values, k=5)

if __name__ == "__main__":
    main()
