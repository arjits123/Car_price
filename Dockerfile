# Setting the base image
FROM python:3.8-slim-buster

# define a working directory
WORKDIR /app 

# copying the entire car_price_prediction folder to working direactory - app
COPY . /app

#updating all the packages
RUN apt update -y && apt install awscli -y

#install requirements.txt
RUN pip install -r requirements.txt
CMD [ "python3","application.py" ]