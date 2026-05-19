# Import/download libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

# Function to preprocess text - take text as input, and return a basic cleaned version of it as output


def preprocess(text) -> str:
    """Take text as input, and return a basic cleaned version of it as output.

    Args:
        text (str): input text/string

    Returns:
        str: cleaned/preprocessed input
    """

    # lowercase everything
    text = text.lower()

    # remove any special and non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)

    # use lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    for i in range(len(tokens)):
        tokens[i] = lemmatizer.lemmatize(tokens[i])
    text = ' '.join(tokens)

    # remove any single letters within a sentence
    text = re.sub(r'(?:^| )\w(?:$| )', ' ', text).strip()

    # remove any word starting with a number
    text = ' '.join(word for word in text.split() if not any(
        letter.isdigit() for letter in word))
    
    # remove numbers in general
    text = ''.join([i for i in text if not i.isdigit()])

    # remove stop words
    text = ' '.join([word for word in text.split()
                    if word not in (stopwords.words('english'))])

    # remove common 'manual' stop words (e.g., 'Swearingen', 'research', 'interests', etc.)
    newStopWords = ['swearingen', 'research', 'interest',
                    'include', 'university', 'south', 'carolina']
    text = ' '.join([word for word in text.split()
                    if word not in newStopWords])

    # return the modified text
    return text


def generate_N_grams(text, ngram=1):
    """Given a text, generate N-grams.

    Args:
        text (str): an input text/string
        ngram (int, optional): size of N. Defaults to 1.

    Returns:
        _type_: N-grams for input text
    """
    try:
        words = [word for word in text.split(" ")]  # split text
    except:
        return []
    temp = zip(*[words[i:] for i in range(0, ngram)])
    ans = [' '.join(ngram) for ngram in temp]
    
    while [] in ans:    # remove extraneous lists (those that are single-word interests, therefore they don't have 2-grams)
        ans.remove([])

    return ans
