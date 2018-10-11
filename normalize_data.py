import pandas as pd
import numpy as np

#importing traing features and labels

read_x = pd.read_csv('train_values.csv')
read_y = pd.read_csv('train_labels.csv')

# assigning patient_id as index for our Data Frame
read_x = read_x.set_index('patient_id')
read_y = read_y.set_index('patient_id')

df = read_x
df = df.join(read_y)   # Add the labels to our data frame to shuffle the data 



df['thal'] = df['thal'].map({'normal': 0, 'reversible_defect':1, 'fixed_defect':2}) # convert string data into integer

# Normalizing the some data ( x- mean(X))/(max-min) to make the features betwwen range -1,1 (this is done for large data only
df['resting_blood_pressure'] = (df['resting_blood_pressure']-
                                np.mean(df['resting_blood_pressure'])) / (np.max(df['resting_blood_pressure'])
                                                                          -np.min(df['resting_blood_pressure']))
df['serum_cholesterol_mg_per_dl'] = (df['serum_cholesterol_mg_per_dl']-
                                      np.mean(df['serum_cholesterol_mg_per_dl'])) / (
                                          np.max(df['serum_cholesterol_mg_per_dl'])-
                                          np.min(df['serum_cholesterol_mg_per_dl']))
df['age'] = (df['age']- np.mean(df['age'])) / (np.max(df['age'])-np.min(df['age']))
df['max_heart_rate_achieved'] = (df['max_heart_rate_achieved']-
                                      np.mean(df['max_heart_rate_achieved'])) / (
                                          np.max(df['max_heart_rate_achieved'])-
                                          np.min(df['max_heart_rate_achieved']))
df = df.sample(frac = 1) #shulling data

df.to_csv('normalized_train_data.csv')

                                 
