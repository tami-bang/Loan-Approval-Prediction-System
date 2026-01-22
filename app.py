# app.py -> loan_prediction.py로 학습된 모델을 predictor.py에서 불러와 사용

from fastapi import FastAPI
from schemas import PredictRequest, PredictResponse
from predictor import predict_loan
from utils import register_exception_handlers

app = FastAPI(title="Loan Approval Prediction API")
register_exception_handlers(app)

@app.get("/")
def root():
    return {"message": "Loan Approval API is running"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):  # /predict POST 요청을 받으면 predictor.py의 predict_loan() 호출
    data = req.dict()
    return predict_loan(data)
