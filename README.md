# Natural-Language-Preprocessing-NLP-(Sentiment Analysis)
Natural Language Preprocessing(NLP) 
#step-1:- Text Preprocessing
Basic text preprocessing.......

for the tex preprocssing fallowa following step
1. Convert all upper case letter into lower cases because python is case sensetive language. For example , Anil and ANIL treats differently


2.#Remove unimportant information (HTML-TAG) , URL and Punctuation.
To remove the HTML tage , make a separate function 

3. Remove URLS for example website link \htttp. or wwww.
  

5. **Spelling correction for text preprocessing.It handel the common types of general mistake....**
   

7.**Emoji Handeling 
use inbuilt deemojization...

8.#**Tokanization very important step of NLP**
why tokanization is so important? It is helps to filter the number of Uniques words. while doing Tokenization alot challenges happen. for example $20 , New-york.....
Tokenization faces challenges such as handling word boundary ambiguity, deciding punctuation inclusion, and addressing complexities in abbreviations and acronyms. Multilingual variations, noisy text, sentence boundary identification, and customization for specific tasks like information extraction further contribute to the nuanced nature of tokenization challenges.

9.**Handalin in tokenization**
 If you're experiencing issues with using the split function in Natural Language Processing (NLP), it's important to understand that the basic split function in Python is limited and may not handle certain tokenization challenges effectively, such as punctuation, contractions, or different languages.
 In NLP, more advanced tokenization methods are often preferred. The nltk library provides a more powerful word_tokenize function that can handle various tokenization challenges.
 Use NLTK inbuilt function. Some time NLTK does not work so try.. Spacy.

 10.**Spacy**
 Spacy is a popular library for natural language processing (NLP) in Python, and it provides robust tools for tokenization and various other NLP tasks. Here's an example of how to use Spacy for tokenization:

 11.**Stemming**
 Stemming is a text normalization process in natural language processing that involves reducing words to their root or base form. It helps in simplifying words to their common base, which can be useful for tasks like information retrieval and text analysis. Here's an example using the NLTK library for stemming:

12.**Snow ball** The Snowball Stemmer is another stemming algorithm, specifically designed to be more aggressive and language-specific than the Porter Stemmer. Here's an example using the Snowball Stemmer in NLTK:

13.**Lemmatization**
Lemmatization is a text normalization process in natural language processing that involves reducing words to their base or dictionary form (lemma). Unlike stemming, lemmatization ensures that the resulting word is a valid word by considering its context and part of speech. Here's an example using the NLTK library for lemmatization:


