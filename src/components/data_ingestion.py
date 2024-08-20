import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from dataclasses import dataclass

#importing from other .py files
from data_transformation import DataTransformation
from model_development import ModelTrainer

# ML libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# DataIngestionConfig will create the input to store csv files
@dataclass #Directly define the class variable, no need to put __init__
class DataIngestionConfig:
    predictor_variable_path: str = os.path.join('artifacts', 'predictors.csv')
    target_variable_path: str = os.path.join('artifacts','target.csv')
    raw_data_path: str = os.path.join('artifacts', 'dataset.csv') # artifacts/raw.csv

#Class to ingest data 
class DataIngestion:
    #Class inheritence
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    #Method to read from data sources
    def initiate_data_ingestion(self):
        logging.info('Entered the Data ingestion method or component')
        try:
            #read dataset
            df = pd.read_csv('notebook/data/cleaned_data.csv') 
            df = df.drop('Unnamed: 0', axis = 1)
            logging.info('Imported cleaned car dataset')

            #Create the artifacts folder, where we have to store train.csv, test and raw
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            #Saving raw.csv file  
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header=True)
            
            #Creating X and y variables
            logging.info('Creating X and y variables')
            predictor_variables = df.drop(columns= ['Price'], axis = 1)
            target_variable = df['Price']

            #Saving train.csv and test.csv
            predictor_variables.to_csv(self.ingestion_config.predictor_variable_path, index = False, header=True)
            target_variable.to_csv(self.ingestion_config.target_variable_path, index = False, header=True)

            logging.info('Data ingestion is completed !')

            return(
                self.ingestion_config.predictor_variable_path,
                self.ingestion_config.target_variable_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':

    #Data Injestion
    data_injestion_obj = DataIngestion()
    predictors, target = data_injestion_obj.initiate_data_ingestion()

    #Data Transformation
    data_transformation_obj = DataTransformation()
    X_train, X_test, y_train, y_test = data_transformation_obj.initiate_data_transformation(predictors, target)

    #Model Development
    model_development_obj = ModelTrainer()
    print(model_development_obj.initiate_model_trainer(X_train,X_test,y_train, y_test))