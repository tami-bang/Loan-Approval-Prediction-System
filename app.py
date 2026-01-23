# app.py
from fastapi import FastAPI
from schemas import PredictRequest, PredictResponse, PredictByLoanIDRequest
from predictor import predict_loan, predict_loan_by_id
from utils import register_exception_handlers
import logging

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI()

# 예외 처리 등록
register_exception_handlers(app)

# 대출 승인 예측 엔드포인트
# POST /predict : feature 기반 대출 승인 예측
@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    # 요청 데이터를 dict로 변환
    data = request.dict()
    logger.info(f"Received request: {data}")
    
    # 모델 예측 호출
    result = predict_loan(data)
    logger.info(f"Returning response: {result}")
    
    # JSON 형태로 반환
    return result

# POST /predict_by_loan_id : Loan_ID 기반 대출 승인 예측
@app.post("/predict_by_loan_id", response_model=PredictResponse)
async def predict_by_loan_id_endpoint(request: PredictByLoanIDRequest):
    logger.info(f"Received Loan_ID request: {request.loan_id}")
    result = predict_loan_by_id(request.loan_id)
    logger.info(f"Returning response: {result}")
    return result
