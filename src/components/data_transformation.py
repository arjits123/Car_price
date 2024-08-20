import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from utils import save_object, TrainTestSplit

from dataclasses import dataclass
# ML Libraries 
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Class to get any required paths for the transformation
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    # feature_engineer_obj_path = os.path.join('artifacts', 'cleaned.pkl')

# Class to transform the data
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def cleaning_dataset(self):
        pass
    
    def get_data_transformation_object(self):
        '''
        This function is responsible for the data transformation
        based on different types of data
        '''
        try:
            categorical_features = ['name', 'company', 'fuel_type']
            numerical_features = ['year', 'kms_driven']
            
            numerical_pipeline = Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean= False))
                ]
            )

            logging.info('Numerical columns standard scaling completed')

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                    ('stdscaler', StandardScaler(with_mean=False))

                ]
            )

            logging.info('Categorical columns encoding completed')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipe', numerical_pipeline, numerical_features),
                    ('categorical_pipe', categorical_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,predictors_path, target_path):
        try:
            #Reading predictor and target data
            X = pd.read_csv(predictors_path)
            y = pd.read_csv(target_path)

            logging.info('Reading X and y variables')

            #Obtaining preprocessing object
            preprocessing_obj = self.get_data_transformation_object()
            logging.info('Reading preprocessing object completed')
            
            # X_train = preprocessing_obj.fit_tranform(X_train)
            predictors_arr = preprocessing_obj.fit_transform(X)
            # print(predictors_arr.shape)

            logging.info('Spliting data into X_train, X_test, y_train, y_test sets')
            #train test split
            X_train, X_test, y_train, y_test = TrainTestSplit(
                X = predictors_arr,
                y = y
            )
            
            # Saving preprocessing object
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info('Saved preprocessing object')

            return(
                X_train,
                X_test,
                y_train,
                y_test
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        