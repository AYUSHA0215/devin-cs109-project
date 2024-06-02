import os
import nltk
import gensim
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re

# Ensure necessary NLTK data packages are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Function to preprocess email text
def preprocess_email(email):
    # Convert to lowercase
    email = email.lower()
    # Remove HTML tags
    email = BeautifulSoup(email, "html.parser").get_text()
    # Remove non-alphabetic characters
    email = re.sub(r'[^a-z\s]', '', email)
    # Tokenize the text
    words = nltk.word_tokenize(email)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return words

# Function to load and preprocess emails from a directory
def load_and_preprocess_emails(directory):
    emails = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='latin-1') as file:
            email = file.read()
            words = preprocess_email(email)
            emails.append(words)
    return emails

def main():
    # Directories containing the email datasets
    easy_ham_dir = 'easy_ham'
    hard_ham_dir = 'hard_ham'
    spam_dir = 'spam'

    # Load and preprocess emails
    easy_ham_emails = load_and_preprocess_emails(easy_ham_dir)
    hard_ham_emails = load_and_preprocess_emails(hard_ham_dir)
    spam_emails = load_and_preprocess_emails(spam_dir)

    # Combine all emails
    all_emails = easy_ham_emails + hard_ham_emails + spam_emails

    # Train Word2Vec model
    model = Word2Vec(sentences=all_emails, vector_size=100, window=5, min_count=1, workers=4)

    # Save the model
    model.save("word2vec.model")

if __name__ == "__main__":
    main()
