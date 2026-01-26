# Loan Approval Prediction API

## 1. Project Overview
본 프로젝트는 학습된 대출 승인 예측 모델을 활용하여  
입력된 고객 정보를 기반으로 대출 승인 여부와 승인 확률을 예측하는  
FastAPI 기반 REST API 서버를 구현하는 것을 목표로 함.

모델은 사전에 학습된 결과를 joblib 파일로 저장하여 사용하며,  
API 서버 시작 시 모델을 로드하여 예측 요청을 처리 함.

# 2. Tech Stack
- Python 3.11.5
- FastAPI
- scikit-learn
- joblib
- Uvicorn

# 3. Project Structure
serving_api/
├─ app.py
├─ predictor.py
├─ schemas.py
├─ utils.py
├─ models/
│ └─ loan_logreg_pipeline_enhanced_py36.joblib
├─ requirements.txt
├─ requirements-train.txt
└─ README.md

# 4. Model
- **Model Type**: Logistic Regression
- **Framework**: scikit-learn
- **Storage**: joblib
- 서버 시작 시 모델을 로드하여 예측에 사용

# 5. API Specification

### POST /predict
대출 신청자의 정보를 입력받아 대출 승인 여부와 승인 확률을 반환

**Request (JSON)**

```json
{
  "gender": 1,
  "married": 0,
  "dependents": 1,
  "education": 1,
  "self_employed": 0,
  "applicant_income": 5000,
  "coapplicant_income": 0,
  "loan_amount": 150,
  "loan_amount_term": 360,
  "credit_history": 1,
  "property_area": 2
}
**Response (JSON)**
{
  "approved": 1,
  "predict_loan": 0.5459
}

# 6. Exception Handling
- Invalid request → 400 Bad Request
- Pydantic 기반 입력 검증을 통해 요청 데이터의 유효성을 검사

# 7. Workflow Diagram
flowchart TD
    A[Client] --> B[POST /predict]
    B --> C[Pydantic Validation]
    C --> D[predictor.predict_loan]
    D --> E[Loaded ML Model (joblib)]
    E --> F[Prediction Result]
    F --> A

# 8. Environment Management
- requirements.txt (serving)
- requirements-train.txt (training)
- 가상환경 디렉터리(.venv*)는 .gitignore를 통해 Git 추적에서 제외

# 9. How to Run
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000

# 10. Test Example
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "gender": 1,
  "married": 0,
  "dependents": 1,
  "education": 1,
  "self_employed": 0,
  "applicant_income": 5000,
  "coapplicant_income": 0,
  "loan_amount": 150,
  "loan_amount_term": 360,
  "credit_history": 1,
  "property_area": 2
}'
