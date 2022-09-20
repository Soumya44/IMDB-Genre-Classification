import pandas as pd
import re
import string
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from joblib import dump, load

lgb_clf = load('./saved_models/lgb_clf.joblib')
wordnet_lemmatizer = WordNetLemmatizer()

def text_cleaner(text):
    # lower-case all characters
    text = text.lower()
    # remove urls
    text =  re.sub(r'http\S+', '',text)
    text =  re.sub(r'www\S+', '',text)                  
    # only keeps characters
    text =  re.sub(r"[^a-zA-Z+]", ' ',text)
    # keep words with length>1 only
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text+' ')
    #remove punctuation marks
    text = "".join([i for i in text if i not in string.punctuation])
    #tokenize
    words = nltk.tokenize.word_tokenize(text)
    #lemmatize
    lemmatizedWords = [wordnet_lemmatizer.lemmatize(i) for i in words]
    #remove stop words
    stopwords = nltk.corpus.stopwords.words('english')
    text = " ".join([i for i in lemmatizedWords if i not in stopwords])
    #strip spaces
    text= re.sub("\s[\s]+", " ",text).strip()
    return text


def get_prediction(text):
    cleanedText = text_cleaner(text)
    ipText = pd.Series([cleanedText])
    pred = lgb_clf.predict(ipText)
    return pred[0]

# print(get_prediction('Thor Love and Thunde'))