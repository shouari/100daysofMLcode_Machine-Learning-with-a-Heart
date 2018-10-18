import pickle
import numpy as np
import pandas as pd
from normalize_data import Normalize
from heart_disease_nn import NeuralNetwork

# Loading data from the csv
#importing traing features and labels

read_x = pd.read_csv('train_values.csv',index_col=0)
read_y = pd.read_csv('train_labels.csv',index_col=0)

df = read_x
df = df.join(read_y)   # Add the labels to our data frame to shuffle the data
#df = df.sample(frac = 1) #shulling data

norm = Normalize(df)
df = Normalize._normalize(norm)

df.to_csv('normalized_train_data.csv')  # Saving the Normalized data frame

dfx = df.drop('heart_disease_present', axis=1)


X = np.array(dfx)

y = df['heart_disease_present'].values
y = y.reshape(180, 1)  # reshaping Y into vector

# The following code's section aims to create 2 classes from our labels ("sick" and Healthy) the
# solution below does look optimum but it is the only one I could find ( If any solution is
# available please let me know

'''label = {'sick': [], 'healthy': []}
for i in range(len(y)):
    if y[i] == 1:
        label['sick'].append(int(0))
        label['healthy'].append(int(1))
    elif y[i] == 0:
        label['sick'].append(int(1))
        label['healthy'].append(int(0))
u = np.array([label['healthy'], label['sick']])
y = u.T '''
m = float(X.shape[0])  # m is the training samples.
training_set = 0.9 # we will use 80% of the data for training while the remain 20%
# will be used for testing '''
m_train = int(m * training_set)
x_train, x_test = X[:m_train, :], X[m_train:, :]
y_train, y_test = y[:m_train], y[m_train:]

# Importing the network for training and setting parameters
# At this stage we can test many many parameters and and select best one


#nn1 = NeuralNetwork(X, y,training_set =training_set, nodes_layer_1=32, nodes_layer_2=32, n_class=1,
                  #epoch=20000, regul_fact=0.4, learning_rate=0.1) #  RELU - Relu Sigmoid
# test_cost for those parameters is 0.4577

nn2 = NeuralNetwork(X, y,training_set =training_set, nodes_layer_1=32, nodes_layer_2=32,
                    epoch=20000, regul_fact=0.5, learning_rate=0.05) #test_cost for those
# parameters is 0.4343

nn2._train_model(x_train, y_train)
nn2._test_model(x_test, y_test)


# once we selected the suitable or best parameters we will save with pickle the model to avoid
# running the training each time we want to make a prediction.
with open('heart_disease_detection_nn2.pickle','wb') as f:
    pickle.dump(nn2,f)
