#Common libraries
import os
import sys
from dataclasses import dataclass
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from utils import save_object, evaluate_model

# Machine learning libraries
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path :str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            # Defining models
            models = {
                'LinearRegression': LinearRegression(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'RandomForestRegressor': RandomForestRegressor()
            }
            
            #Imported evaluate_model() function from utils
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)

            #To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            #Get the best model name 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            #Get best model name
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info('Best found model on test dataset')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            #Prediction of best model
            predicted = best_model.predict(X_test)
            score = r2_score(y_test,predicted)
            return best_model_name, score

        except Exception as e:
            raise CustomException(e,sys)
        

        
