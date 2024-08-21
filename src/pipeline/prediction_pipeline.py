import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:

            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            scaled_data = preprocessor.transform(features)
            prediction = model.predict(scaled_data)
            return prediction
        except Exception as e:
            raise CustomException(e,sys)



"""
Below class is responsible for mapping all the inputs we are giving in HTML to the backend with these below values
"""
class CustomData:
    def __init__(self, name:str, company:str, year:int, kms_driven:int,fuel_type:str):
        self.name = name
        self.company = company
        self.year = year
        self.kms_driven = kms_driven
        self.fuel_type = fuel_type

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "name": [self.name],
                "company": [self.company],
                "year": [self.year],
                "kms_driven": [self.kms_driven],
                "fuel_type": [self.fuel_type]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)

        