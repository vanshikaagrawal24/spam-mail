import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

data = pd.read_csv('mail_data.csv')

data.isnull().sum()
mail_data = data.where((pd.notnull(data)), '')

mail_data.loc[mail_data['Category'] == 'spam','Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham','Category'] = 1

Y = mail_data['Category']
X = mail_data['Message']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state =3)

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

model = LogisticRegression()
model.fit(X_train_features, Y_train)

X_train_prediction = model.predict(X_train_features)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = model.predict(X_test_features)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print(training_data_accuracy)

print(test_data_accuracy)

input_data = ["Fair enough, anything going on?"]
input_data_features = feature_extraction.transform(input_data)
prediction = model.predict(input_data_features)
print(prediction)

pickle.dump(model,open('spam.pkl','wb'))
pickle.dump(feature_extraction, open('features.pkl', 'wb'))