import pandas as pd
from normalize_data import Normalize
import numpy as np
from heart_disease_nn import NeuralNetwork
import pickle

#importing Submission data set

read_x = pd.read_csv('test_values.csv', index_col = 0)
df = read_x

# Normalize submission data
norm = Normalize(df)
df = Normalize._normalize(norm)
# converting data frame into array
x_submission = np.array(df)

# Loading the trained model
pickle_in = open('heart_disease_detection_nn2.pickle','rb')
model = pickle.load(pickle_in)
prediction = model._predict(x_submission)[:,0]
prediction = np.around(prediction, decimals=2)

df['heart_disease_present'] = prediction
submission = df['heart_disease_present']
submission.to_csv("submission-1.csv", header=[ 'heart_disease_present'])
