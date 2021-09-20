import nltk
# nltk.download('punkt')
import numpy as np

from nltk.stem.porter import PorterStemmer
stemmer =PorterStemmer()

def tokenize(sentence):
    """
        split sentence into array of words/tokens
        a token can be a word or punctuation character, or number
        """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
        stemming = find the common word
        examples:
        words = ["organize", "organizes", "organizing"]
        words = [stem(w) for w in words]
        -> ["organ", "organ", "organ"]
        """
    return stemmer.stem(word.lower())

# a ="How long does it takes for shipping"
# print(a)
# a = tokenize(a)
# print(a)
#
# words= ["Organise","organiser","organising"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)
# a =stem(a)
# print(a)

def bag_of_words(tokenized_sentence, allword):
    """
        return bag of words array:
        1 for each known word that exists in the sentence, 0 otherwise
        example:
        words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        sentence = ["hello", "how", "are", "you"]
        bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
        """
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(allword), dtype=np.float32)
    for idx, w in enumerate(allword):
        if w in tokenized_sentence:
            bag[idx] = 1

    return bag