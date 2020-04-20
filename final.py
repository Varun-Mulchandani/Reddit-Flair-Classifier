from flask import Flask,render_template,url_for,request
import urllib.request, json
import os
import pickle
#from tensorflow import set_random_seed
#from numpy.random import seed
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

train = pd.read_csv('reddddddit.csv')
max_no_word = 50000
max_seq_length = 250
embedding_dim = 100

tokenizer = Tokenizer(num_words = max_no_word, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower = True)
tokenizer.fit_on_texts(train['title'].values)
list_classes = ["AskIndia", "Non-Political", "[R]eddiquette",
                "Scheduled", "Photography", "Science/Technology",
                "Politics", "Business/Finance", "Policy/Economy",
                "Sports", "Food", "AMA"]
def generate_flair(title):
    title = [title]
    seq = tokenizer.texts_to_sequences(title)
    padded = pad_sequences(seq, maxlen = max_seq_length)
    pred = model.predict(padded)
    return list_classes[np.argmax(pred)]


app = Flask(__name__)
@app.route('/')
def main():
    return render_template('main.html')
@app.route('/predict',methods=['POST'])
def predict():
    global model
    #set_random_seed(2)
    #seed(1)
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action = 'ignore',category = FutureWarning)
    model = pickle.load(open('Classifier_bert.pkl','rb'))

    if request.method == 'POST':
        message = request.form['message']
        url = message + '.json'
        with urllib.request.urlopen(url) as url:
            data = json.loads(url.read().decode())
        title = data[0]['data']['children'][0]['data']['title']
        my_prediction = generate_flair(title)
#print(my_prediction)
#my_prediction = model.predict()
    return render_template('mainresult.html',prediction = my_prediction)

if __name__ == '__main__':
  app.run(debug = True, use_reloader = False)
