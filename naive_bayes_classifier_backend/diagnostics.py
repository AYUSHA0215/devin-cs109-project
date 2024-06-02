import pickle
import numpy as np

# Load the vocabulary and model parameters from files
with open('vocabulary.pkl', 'rb') as vocab_file:
    vocabulary = pickle.load(vocab_file)

priors = np.load('priors.npy')
likelihoods = np.load('likelihoods.npy')

# Print the sizes of the loaded vocabulary and the likelihoods matrix
print(f"Loaded vocabulary size: {len(vocabulary)}")
print(f"Likelihoods shape: {likelihoods.shape}")

# Additional diagnostic print statements for email BoW and TF-IDF shapes
sample_email = "This is a test email to check the classification functionality."
preprocessed_email = preprocess_email(sample_email)
email_bow = transform_to_bow([preprocessed_email], vocabulary)
email_tfidf = calculate_tfidf(email_bow)

print(f"email_bow shape: {email_bow.shape}")
print(f"email_tfidf shape: {email_tfidf.shape}")