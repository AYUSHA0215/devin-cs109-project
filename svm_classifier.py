import numpy as np
from preprocess_emails import load_and_preprocess_emails, transform_to_word2vec, train_test_split, calculate_metrics
from gensim.models import Word2Vec

class SVMClassifier:
    def __init__(self, learning_rate=0.01, lambda_param=0.1, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (x_i.dot(self.w) + self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w += self.learning_rate * np.multiply(x_i.reshape(-1), y_[idx])
                    self.b += self.learning_rate * y_[idx]

    def predict(self, X):
        approx = X.dot(self.w) + self.b
        predictions = np.sign(approx)
        return predictions

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

    # Load the Word2Vec model
    word2vec_model = Word2Vec.load("word2vec.model")

    # Transform emails to Word2Vec vectors
    email_vectors = transform_to_word2vec(all_emails, word2vec_model)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(email_vectors, labels)

    # Train the SVM classifier
    svm = SVMClassifier()
    svm.fit(X_train, y_train)

    # Predict using the SVM classifier
    y_pred = svm.predict(X_test)

    # Calculate and print evaluation metrics
    accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
    print(f"SVM Classifier - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

if __name__ == "__main__":
    main()
