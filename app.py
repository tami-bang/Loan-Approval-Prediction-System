# app.py

from fastapi import FastAPI
from schemas import PredictRequest, PredictResponse
from predictor import predict_loan
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
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    # 요청 데이터를 dict로 변환
    data = request.dict()
    logger.info(f"Received request: {data}")
    
    # 모델 예측 호출
    result = predict_loan(data)
    logger.info(f"Returning response: {result}")
    
    # JSON 형태로 반환
    return result

