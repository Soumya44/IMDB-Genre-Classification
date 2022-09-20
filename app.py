from crypt import methods
from unittest import result
from flask import Flask, request, render_template

import pandas as pd
import re
import string
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC,LinearSVC
import lightgbm as lgb

import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import Callback, EarlyStopping
import pickle


from sklearn.pipeline import Pipeline
from joblib import dump, load


svc_clf = load('./saved_models/svc_clf.joblib')
lgb_clf = load('./saved_models/lgb_clf.joblib')

tokenizer = None
labels = None
with open('./saved_models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('./saved_models/labels.pickle', 'rb') as handle:
    labels = pickle.load(handle)

EMBEDDING_DIM = 32
model = Sequential()
model.add(Embedding(50000, EMBEDDING_DIM, input_length=200))
model.add(LSTM(100, dropout=0.1))
model.add(Dense(27, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights('./saved_models/lstm_checkpoint')

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



app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', selected_svc = 'selected')

@app.route('/predict', methods=['POST'])
def predict():
    
    error = None
    result = None
    clf=None
    selected_svc = ''
    selected_lgb = ''
    selected_lstm = ''

    text = request.form['Description']
    classifier = request.form['Classifier']

    if classifier == 'svc':
        clf = svc_clf
        cleanedText = text_cleaner(text)
        ipText = pd.Series([cleanedText])
        pred = clf.predict(ipText)
        result = pred[0]
        selected_svc = 'selected'
        selected_lgb = ''
        selected_lstm = ''
    elif classifier == 'lgb':
        clf = lgb_clf
        cleanedText = text_cleaner(text)
        ipText = pd.Series([cleanedText])
        pred = clf.predict(ipText)
        result = pred[0]
        selected_lgb = 'selected'
        selected_svc = ''
        selected_lstm = ''
    elif classifier == 'lstm':
        cleanedText = text_cleaner(text)
        ipText = pd.Series([cleanedText])
        ip = tokenizer.texts_to_sequences(ipText.values)
        ip = pad_sequences(ip, maxlen=200)
        pred = model.predict(ip)
        res = np.argmax(pred,axis=1)
        result = labels[res[0]]
        selected_lgb = ''
        selected_svc = ''
        selected_lstm = 'selected'


    return render_template('index.html', result = result, text = text, prediction_success = True, selected_lgb = selected_lgb,  selected_lstm = selected_lstm, selected_svc = selected_svc)


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=3000, debug=True)