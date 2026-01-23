# schemas.py
from pydantic import BaseModel, Field, conint, confloat, constr

# 요청 데이터 구조 정의
class PredictRequest(BaseModel):
    # 성별
    Gender: constr(strip_whitespace=True)
    # 결혼 여부
    Married: constr(strip_whitespace=True)
    # 부양 가족 수, 0 이상
    Dependents: conint(ge=0) = Field(..., description="Number of dependents, must be >= 0")
    # 학력
    Education: constr(strip_whitespace=True)
    # 자영업 여부
    Self_Employed: constr(strip_whitespace=True)
    # 주거 지역
    Property_Area: constr(strip_whitespace=True)
    # 대출 금액, 0보다 큰 값
    LoanAmount: confloat(gt=0) = Field(..., description="Loan amount must be > 0")
    # 대출 기간, 0보다 큰 값
    Loan_Amount_Term: conint(gt=0) = Field(..., description="Term must be > 0")
    # 총 소득, 0보다 큰 값
    TotalIncome: confloat(gt=0) = Field(..., description="Total income must be > 0")
    # 신용 이력, 0 또는 1
    Credit_History: conint(ge=0, le=1) = Field(..., description="0 or 1")
    # 부양 가족당 소득, 0 이상
    IncomePerDependent: confloat(ge=0)
    # 대출 금액 대비 소득 비율, 0 이상
    LoanAmountToIncome: confloat(ge=0)
    # 학력과 자영업 결합 필드
    Education_SelfEmployed: constr(strip_whitespace=True)

# Loan_ID 기반 예측 요청 데이터 구조 정의
class PredictByLoanIDRequest(BaseModel):
    # Loan_ID 기준 예측 요청
    loan_id: str = Field(..., description="Loan_ID 기준 예측")

# 응답 데이터 구조 정의
class PredictResponse(BaseModel):
    # 대출 승인 여부, 0 또는 1
    approved: int
    # 대출 승인 확률
    predict_loan: float

