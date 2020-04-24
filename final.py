#importing the necessary libraries

from flask import Flask,render_template,url_for,request,jsonify
from flask_restful import reqparse, abort, Api, Resource
import urllib.request, json
import os
import pickle
import praw
import nltk
from nltk.corpus import stopwords
import regex as re
import pandas as pd
import numpy as np
import string, os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout

#Reading the data

train = pd.read_csv('reddddddit.csv')
max_no_word = 50000
max_seq_length = 250
embedding_dim = 100

#Creating the tokenizer

tokenizer = Tokenizer(num_words = max_no_word, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower = True)
tokenizer.fit_on_texts(train['title'].values)

#Using praw to get data of the submission via the url

reddit = praw.Reddit(client_id ='Q99qSQY6otSnWw',
                     client_secret ='LrcqTOgL_perr75LA2n0WpDTa3A',
                     user_agent = 'reddit_scraper' )
list_classes = ["AskIndia", "Non-Political" ,
                "Scheduled", "Photography", "Science/Technology",
                "Politics", "Business/Finance", "Policy/Economy",
                "Sports", "Food", "Coronavirus"]

#Fucntion to predict flair from given title

def generate_flair(title):
    title = [title]
    seq = tokenizer.texts_to_sequences(title)
    padded = pad_sequences(seq, maxlen = max_seq_length)
    pred = model.predict(padded)
    return list_classes[np.argmax(pred)]

#Function to extarct the urls from the text file

def extract(input_url):
    input_title = input_url.split("/")
    if input_title[-1].isalpha():
        id_val = input_title[-2]
    else:
        id_val = input_title[-3]
    return str(id_val)

#Function to clean the data by removing unwanted symbols and stop words

replace = re.compile('[/(){}\[\]\|@,;]')
symbols = re.compile('[^0-9a-z #+_]')
stopword_s = set(stopwords.words('english'))

def clean(title):
    title = title.lower()
    title = replace.sub(' ', title)
    title = symbols.sub('', title)
    title = ' '.join(word for word in title.split() if word not in stopword_s)
    return title

#app

app = Flask(__name__)
@app.route('/')
def main():
    return render_template('main.html')

#For displaying predicted value
@app.route('/predict',methods=['POST'])
def predict():
    global model
    #set_random_seed(2)
    #seed(1)
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action = 'ignore',category = FutureWarning)
    model = pickle.load(open('lstm_model.pkl','rb'))

    if request.method == 'POST':
        message = request.form['message']
        #url = message + '.json'
        #with urllib.request.urlopen(url) as url:
        #    data = json.loads(url.read().decode())
        #title = data[0]['data']['children'][0]['data']['title']
        #my_prediction = generate_flair(title)
        submission = reddit.submission(url = message)
        title = str(submission.title)
        title = clean(title)
        my_prediction = generate_flair(title)
    return render_template('mainresult.html',prediction = my_prediction)

#For automated testing

@app.route('/automated_testing', methods=['GET', 'POST'])
def automated_testing():
    global model
    file = request.files['file']
    content = file.readlines()
    content = [x.strip() for x in content]
    values = {}
    model = pickle.load(open('lstm_model.pkl','rb'))
    for x in content:
        x = x.decode()
        id_val = extract_id(x)
        sub = reddit.submission(id=id_val)
        title = sub.title
        title = clean(title)
        prediction = generate_flair(title)
        values.update({x:prediction})
    return jsonify(values)
    if request.method == 'GET':
        return "Awaiting POST request."

if __name__ == '__main__':
  app.run(debug = True, use_reloader = False)
