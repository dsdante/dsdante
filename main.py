import io
import pickle
from typing import Iterable

import pandas as pd
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import skab_pytorch

CSV_HEADER = ("datetime;Accelerometer1RMS;Accelerometer2RMS;Current;"
              "Pressure;Temperature;Thermocouple;Voltage;Volume Flow RateRMS")

app = FastAPI()


class Model(BaseModel):
    X: list[str]


@app.get('/', summary="Redirects to Swagger.")
def root():
    return RedirectResponse('docs')


@app.post("/predict", summary="Accepts a list of CSV lines (without a header).")
def predict(model: Model):
    string_data = CSV_HEADER + '\n' + '\n'.join(model.X)
    df = pd.read_csv(io.StringIO(string_data), sep=';', index_col='datetime', parse_dates=True)
    anomalies = skab_pytorch.predict_df(df)
    return {'anomalies': ''.join(map(str, anomalies))}


@app.post("/predict/file", summary="Accepts *.csv files and Pickle binaries.")
def predict(file: UploadFile):
    if file.filename.lower().endswith('csv'):
        df = pd.read_csv(file.file, sep=';', index_col='datetime', parse_dates=True)
    else:
        df = pickle.load(file.file)
        if isinstance(df, Iterable):
            df = pd.concat(df)
    anomalies = skab_pytorch.predict_df(df)
    return {'anomalies': ''.join(map(str, anomalies))}


if __name__ == '__main__':
    uvicorn.run('__main__:app', host='0.0.0.0', port=8000, reload=True)
