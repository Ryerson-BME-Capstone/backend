from fastapi import FastAPI
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model

app = FastAPI()

model = load_model('prediction')


@app.post('/prediction' )
def get_potability(dataframe: df):
    data = dataframe.iloc[1:, 1:].astype(np.float).T
    data = dataframe.dropna()
    y = model.predict(data)
    y = [0 if val < 0.5 else 1 for val in y]
    for y = 1:
        prediction="You will survive."
    for y = 0:
        prediction="You will not survive."
    return prediction

