import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """Tokenizes a sentence into words."""
    return nltk.word_tokenize(sentence)

def stem(word):
    """Applies stemming to the word to get its root form."""
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """Creates a bag of words representation for a sentence."""
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    
    for idx, word in enumerate(words):
        if word in sentence_words:
            bag[idx] = 1
    
    return bag