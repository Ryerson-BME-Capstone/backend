from fastapi import FastAPI
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model

app = FastAPI()
model = load_model('prediction')

def match(seq1, seq2):
    """Finds the index locations of seq1 in seq2"""
    return [ np.nonzero(seq2==x)[0][0] for x in seq1  if x in seq2 ]


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

