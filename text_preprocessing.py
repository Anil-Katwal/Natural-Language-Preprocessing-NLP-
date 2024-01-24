# Text_Preprocessing for the NLP.......................
#prepare by Anil Katwal

import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

!pip install gensim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gensim
from nltk import sent_tokenize
from gensim.utils import simple_preprocess

df = pd.read_csv('Film_IBMDATA_set', nrows=50000)  # Adjust the number of rows as needed
df.head()

"""First step is converting all text into lower casing bold text since python is case sensative"""

df['review'][3].lower()

df.shape

"""**To convert all review into the lower cases.**"""

df['review'].str.lower()

df['review']=df['review'].str.lower()

df['review']

"""**Remove unimportant information(HTML TAG) use this function to rem0ve html**"""

import re

def html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub('', text)

df['review_without_html'] = df['review'].apply(html_tags)
print(df[['review', 'review_without_html']].head())

"""**Remove URLS for example whatsapp data or chat or wikipedia text**"""

import re

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)
text_with_urls = "Check out my website at https://www.example.com. For more information, visit http://anotherexample.com."
text_without_urls = remove_urls(text_with_urls)

print("Original text:", text_with_urls)
print("Text without URLs:", text_without_urls)

"""**Remove Punctuations**"""

import string
import time


punctuation_chars = string.punctuation
print(punctuation_chars)

exclude=string.punctuation

import re
import string

def punctuation_chars(text, exclude=None):
    if exclude is None:
        exclude = string.punctuation

    for char in exclude:
        text = text.replace(char, '')

    return text

# Example usage:
text_with_punctuation = "Hello, world! This is an example text with punctuation."
text_without_punctuation = punctuation_chars(text_with_punctuation)

print("Original text:", text_with_punctuation)
print("Text without punctuation:", text_without_punctuation)

start=time.time()
print(punctuation_chars)
time1=time.time()-start
print(time1)

"""**This is too slow for the fast processing use following functions to remove function.....**"""

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

start=time.time()
print(punctuation_chars)
time2=time.time()-start
print(time2)

df['review'].apply(remove_punctuation)

"""**Chat word treatment or Salang word treatment** for this you have to used dictionary........... Use following function"""

def chat_conversation(text):
    new_text = []
    chat_words = {"R": "are", "U": "you", "L8": "late", "BRB": "be right back"}

    for word in text.split():
        if word.upper() in chat_words:
            new_text.append(chat_words[word.upper()])
        else:
            new_text.append(word)

    return ' '.join(new_text)

# Example usage:
input_text = "R U coming or not? BRB, I'm L8."
output_text = chat_conversation(input_text)

print("Original text:", input_text)
print("Converted text:", output_text)

"""**Spelling correction for text preprocessing.It handel the common types of general mistake....**"""

from textblob import TextBlob
text_with_mistakes = "Thee quick brown fox jumpd ovver the lazy dog."
blob = TextBlob(text_with_mistakes)
corrected_text = blob.correct()

print("Original Text:", text_with_mistakes)
print("Corrected Text:", corrected_text)

"""**Remove the stop words**"""

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print(stop_words)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
# This function remove the stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)

    return filtered_text

# Example usage:
input_text = "This is an example sentence with some stopwords."
output_text = remove_stopwords(input_text)

print("Original Text:", input_text)
print("Text without Stopwords:", output_text)

"""**Emoji Handeling Function.....** use the demojize function....."""

!pip install emoji

import emoji

def remove_emojis(text):
    cleaned_text = emoji.demojize(text)
    cleaned_text = cleaned_text.replace(":", "")
    return cleaned_text

# Example usage:
text_with_emojis = "Hello! 😊 How are you today? 🌟"
text_without_emojis = remove_emojis(text_with_emojis)

print("Original Text:", text_with_emojis)
print("Text without Emojis:", text_without_emojis)

"""**Tokanization very important step of NLP**
why tokanization is so important? It is helps to filter the number of Uniques words. while doing Tokenization alot challenges happen. for example $20 , New-york.....
Tokenization faces challenges such as handling word boundary ambiguity, deciding punctuation inclusion, and addressing complexities in abbreviations and acronyms. Multilingual variations, noisy text, sentence boundary identification, and customization for specific tasks like information extraction further contribute to the nuanced nature of tokenization challenges.
"""

#1 use split function
def tokenize_with_split(text):
    # Tokenize using split function
    tokens = text.split()
    return tokens

# Example usage:
example_text = "Tokenization using split function is a basic approach."
tokens = tokenize_with_split(example_text)

print("Original Text:", example_text)
print("Tokens using split:", tokens)

"""**Handalin in tokenization**
 If you're experiencing issues with using the split function in Natural Language Processing (NLP), it's important to understand that the basic split function in Python is limited and may not handle certain tokenization challenges effectively, such as punctuation, contractions, or different languages.
 In NLP, more advanced tokenization methods are often preferred. The nltk library provides a more powerful word_tokenize function that can handle various tokenization challenges  **NLTK**
"""

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def tokenize_with_nltk(text):
    # Tokenize using nltk's word_tokenize
    tokens = word_tokenize(text)
    return tokens
# Example usage:
example_text = "Advanced NLP tokenization requires specialized tools?But I am going to new-york."
tokens_nltk = tokenize_with_nltk(example_text)

print("Original Text:", example_text)
print("Tokens using nltk:", tokens_nltk)

"""**Spacy**
Spacy is a popular library for natural language processing (NLP) in Python, and it provides robust tools for tokenization and various other NLP tasks. Here's an example of how to use Spacy for tokenization:
"""

import spacy


nlp = spacy.load("en_core_web_sm")
def tokenize_with_spacy(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]

    return tokens

# Example usage:
example_text = "I am Ph.D student."
tokens_spacy = tokenize_with_spacy(example_text)

print("Original Text:", example_text)
print("Tokens using Spacy:", tokens_spacy)

"""**Stemming**
Stemming is a text normalization process in natural language processing that involves reducing words to their root or base form. It helps in simplifying words to their common base, which can be useful for tasks like information retrieval and text analysis. Here's an example using the NLTK library for stemming:
"""

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
porter_stemmer = PorterStemmer()

def stem_text(text):
    words = word_tokenize(text)
    stemmed_words = [porter_stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)

    return stemmed_text

# Example usage:
original_text = "I am very interested in the NPL algorithm and i want to be very successful data scientist and ML engineer."
stemmed_text = stem_text(original_text)

print("Original Text:", original_text)
print("Stemmed Text:", stemmed_text)

"""**Snow ball** The Snowball Stemmer is another stemming algorithm, specifically designed to be more aggressive and language-specific than the Porter Stemmer. Here's an example using the Snowball Stemmer in NLTK:"""

from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
# Initialize the Snowball Stemmer for English
snowball_stemmer = SnowballStemmer('english')
def snowball_stem_text(text):
    words = word_tokenize(text)
    stemmed_words = [snowball_stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text

# Example usage:
original_text = "Stemming with Snowball is more aggressive and language-specific."
stemmed_text_snowball = snowball_stem_text(original_text)

print("Original Text:", original_text)
print("Snowball Stemmed Text:", stemmed_text_snowball)

"""**Lemmatization** Lemmatization is a text normalization process in natural language processing that involves reducing words to their base or dictionary form (lemma). Unlike stemming, lemmatization ensures that the resulting word is a valid word by considering its context and part of speech. Here's an example using the NLTK library for lemmatization:"""

import nltk

# Download the WordNet resource
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_text = ' '.join(lemmatized_words)

    return lemmatized_text

# Example usage:
original_text = "I am going to University and talk with my professor for my ph.d research."
lemmatized_text = lemmatize_text(original_text)

print("Original Text:", original_text)
print("Lemmatized Text:", lemmatized_text)
