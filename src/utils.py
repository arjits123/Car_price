import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging

import numpy as np
import pandas as pd
import pickle
import dill
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def TrainTestSplit(X,y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=124)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(models)):
            #list of all the models
            model = list(models.values())[i]

            #train the model
            model.fit(X_train,y_train)
            #make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            #evaluate the model
            training_score = r2_score(y_train, y_train_pred)
            testing_score = r2_score(y_test, y_test_pred)
            
            #report[keys] = value
            report[list(models.keys())[i]] = testing_score
        
        return report
    except Exception as e:
        raise CustomException(e,sys)
