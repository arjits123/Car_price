from flask import Flask
from flask import request
from flask import render_template

import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

# calling the dataset
car = pd.read_csv('notebook/data/cleaned_data.csv')

@app.route('/')
def index():
    company = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    return render_template('index.html', companies = company, car_models = car_models, years = year, fuel_types = fuel_type)

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data = CustomData(
            name = request.form.get('car_models'),
            company = request.form.get('company'),
            year = float(request.form.get('years')),
            kms_driven = float(request.form.get('kilo_driven')),
            fuel_type = request.form.get('fuel_types')
        )

        prediction_df = data.get_data_as_data_frame()
        print(prediction_df)
        
        #Initialise my pipeline
        prediction_pipline = PredictPipeline()
        results = prediction_pipline.predict(prediction_df)
        return render_template('index.html', results = np.round(results[0][0]))

if __name__ == '__main__':
    app.run()