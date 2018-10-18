import pandas as pd
import numpy as np



class Normalize:
    def __init__(self, data_frame):
       self.data_frame = data_frame

    def _normalize(self):
        df = self.data_frame
        df['thal'] = df['thal'].map({'normal': 0, 'reversible_defect':1, 'fixed_defect':2}) #
        # convert string data into integer

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
        return df

