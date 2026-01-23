# app.py
from fastapi import FastAPI
from schemas import PredictRequest, PredictResponse, PredictByLoanIDRequest
from predictor import predict_loan, predict_loan_by_id
from utils import register_exception_handlers
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
import logging
import os

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI()

# 예외 처리 등록
register_exception_handlers(app)

# === 정적 파일 설정 ===
# 프로젝트 루트에 static/ 폴더가 있어야 함
if not os.path.exists("static"):
    os.mkdir("static")

# /static 경로로 css, js, img, lib 파일 접근 가능
app.mount("/static", StaticFiles(directory="static"), name="static")

# 루트 접근 시 SB Admin 2 템플릿으로 리다이렉트
@app.get("/")
async def root():
    # 브라우저에서 / 접속 시 SB Admin 2 index.html로 이동
    return RedirectResponse(url="/static/sb-admin-2/index.html")

# === API 엔드포인트 ===

# POST /predict : feature 기반 대출 승인 예측
@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    data = request.dict()
    logger.info(f"Received request: {data}")
    result = predict_loan(data)
    logger.info(f"Returning response: {result}")
    return result

# POST /predict_by_loan_id : Loan_ID 기반 대출 승인 예측
@app.post("/predict_by_loan_id", response_model=PredictResponse)
async def predict_by_loan_id_endpoint(request: PredictByLoanIDRequest):
    logger.info(f"Received Loan_ID request: {request.loan_id}")
    result = predict_loan_by_id(request.loan_id)
    logger.info(f"Returning response: {result}")
    return result
