# schemas.py -> 모델 학습 완료 (이미 joblib 저장)

from pydantic import BaseModel

# 응답 데이터 구조
class PredictRequest(BaseModel):  # 2-(2,3) 충족
    Gender: str
    Married: str
    Dependents: int
    Education: str
    Self_Employed: str
    Property_Area: str
    LoanAmount: float
    Loan_Amount_Term: float
    TotalIncome: float
    Credit_History: float
    IncomePerDependent: float
    LoanAmountToIncome: float
    Education_SelfEmployed: str

# 응답 데이터 구조
class PredictResponse(BaseModel):  # 3-(2,3) 충족
    approved: int
    predict_loan: float
