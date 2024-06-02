from gensim.models import KeyedVectors

try:
    model = KeyedVectors.load('word2vec.model')
    print("Model loaded successfully.")
except Exception as e:
    print("An error occurred while loading the model:")
    print(e)
