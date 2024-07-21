from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__, template_folder='template')

model = pickle.load(open('spam.pkl', 'rb'))

feature_extraction = pickle.load(open('features.pkl', 'rb'))

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        mail = request.form['mail']
        input_data = [mail]
        input_data_features = feature_extraction.transform(input_data)
        prediction = model.predict(input_data_features)
        if(prediction==0):
            return render_template('index.html', result = "Spam Mail detected")
        else:
            return render_template('index.html', result = "Ham Mail detected")


if __name__ == '__main__':
    app.run(debug=True)