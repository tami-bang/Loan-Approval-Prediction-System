# app/main.py
from fastapi import FastAPI
from app.utils import register_exception_handlers, log_request_time, health_check, logger
from predictor import predict_loan, predict_loan_by_id
from schemas import PredictRequest, PredictResponse, PredictByLoanIDRequest
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings


# FastAPI 앱 생성, 제목과 버전 설정
app = FastAPI(title="Loan Approval Prediction API", version="1.0.0")

# CORS 미들웨어 등록
# 모든 도메인에서 요청 허용, 크레덴셜 포함, 모든 메서드/헤더 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 처리 시간 로깅 미들웨어 등록
app.middleware("http")(log_request_time)

# 커스텀 예외 핸들러 등록
register_exception_handlers(app)

# 헬스 체크 엔드포인트
# 서버 상태 확인용, 필요시 DB/모델 상태도 추가 가능
@app.get("/health")
def health():
    return health_check()

# 대출 예측 엔드포인트 (전체 feature 기반)
# PredictRequest 모델로 입력 받고 PredictResponse 모델로 반환
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # 요청 데이터를 dict로 변환하여 predict_loan 호출
    approved, predict_loan_val = predict_loan(request.dict())
    # 예측 결과 로그 기록
    logger.info(f"Predict result: approved={approved}, predict_loan={predict_loan_val}")
    # 승인 여부와 예측 확률 반환
    return {"approved": approved, "predict_loan": predict_loan_val}

# 대출 예측 엔드포인트 (Loan_ID 기반)
# PredictByLoanIDRequest 모델로 입력 받고 PredictResponse 모델로 반환
@app.post("/predict_by_loan_id", response_model=PredictResponse)
def predict_by_loan_id(request: PredictByLoanIDRequest):
    # Loan_ID 기반으로 predict_loan_by_id 호출
    approved, predict_loan_val = predict_loan_by_id(request.loan_id)
    # 예측 결과 로그 기록
    logger.info(f"Predict by Loan_ID result: Loan_ID={request.loan_id}, approved={approved}, predict_loan={predict_loan_val}")
    # 승인 여부와 예측 확률 반환
    return {"approved": approved, "predict_loan": predict_loan_val}
