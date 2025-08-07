# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List
# import joblib
# import numpy as np

# app = FastAPI(title="SilverBird API")

# model = joblib.load("model/stock_predictor.pkl")

# class InputData(BaseModel):
#     features:List

# @app.post("/predict")
# def predict(data:InputData)
#     try:
#         prediction = model.predict([data.features])
#         return {"prediction": float(prediction[0])}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))