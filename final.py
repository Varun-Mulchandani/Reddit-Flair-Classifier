from flask import Flask,render_template,url_for,request
import urllib.request, json
from sklearn.externals import joblib
import os
from tensorflow import set_random_seed
from numpy.random import seed
import pandas as pd
import numpy as np
import string, os
import warnings


def generate_flair(sentence):
  all_tokens = []
  all_masks = []
  all_segments = []
  sentence = tokenizer.tokenize(sentence)
  sentence = sentence[:160 - 2]
  input_sequence = ['[CLS]'] + sentence + ['[SEP]']
  pad_len = 160 - len(input_sequence)
  tokens = tokenizer.convert_tokens_to_ids(input_sequence)
  tokens += [0]* pad_len
  pad_masks = [1] * len(input_sequence) + [0] * pad_len
  segment_ids = [0] * 160
  all_tokens.append(tokens)
  all_masks.append(pad_masks)
  all_segments.append(segment_ids)
  input_s = (np.array(all_tokens), np.array(all_masks), np.array(all_segments))
  out = model.predict(input_s)
  out.argmax()
  for name, ids in label_to_id.items():
    if ids == (out.argmax()):
      finaloutput = name
  return finaloutput


app = Flask(__name__)
@app.route('/')
def main():
    return render_template('main.html')
@app.route('/predict',methods=['POST'])
def predict():
    set_random_seed(2)
    seed(1)
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action = 'ignore',category = FutureWarning)
    Classifier_bert = open('Classifier_bert.pkl','rb')
    model = joblib.load(Classifier_bert)

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
