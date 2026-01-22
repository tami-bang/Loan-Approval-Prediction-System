# predictor.py

import os
import joblib
import pandas as pd
import traceback

# 모델 파일 경로
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "loan_logreg_pipeline_enhanced_py36.joblib")

# 모델 로드
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# 모델이 기대하는 컬럼 순서
FEATURE_COLUMNS = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "Property_Area", "LoanAmount", "Loan_Amount_Term", "TotalIncome",
    "Credit_History", "IncomePerDependent", "LoanAmountToIncome",
    "Education_SelfEmployed"
]

# 수치형 컬럼
NUMERIC_COLUMNS = [
    "Dependents", "LoanAmount", "Loan_Amount_Term", "TotalIncome",
    "Credit_History", "IncomePerDependent", "LoanAmountToIncome"
]

# 범주형 컬럼
CATEGORICAL_COLUMNS = [
    "Gender", "Married", "Education", "Self_Employed",
    "Property_Area", "Education_SelfEmployed"
]

def predict_loan(features: dict) -> dict:
    try:
        # 1. FEATURE_COLUMNS 순서대로 데이터 추출
        row = {col: features[col] for col in FEATURE_COLUMNS}

        # 2. 수치형 컬럼 float로 변환
        for col in NUMERIC_COLUMNS:
            try:
                row[col] = float(row[col])
            except Exception as e:
                raise ValueError(f"Numeric conversion failed for column '{col}': {row[col]}")

        # 3. 범주형 컬럼 str로 변환
        for col in CATEGORICAL_COLUMNS:
            row[col] = str(row[col])

        # 4. DataFrame 생성
        X = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        # 5. 디버깅 출력: DataFrame 내용과 컬럼별 dtype
        print("=== DEBUG: DataFrame for prediction ===")
        print(X)
        print("=== DEBUG: dtypes ===")
        print(X.dtypes)

        # 6. 예측 수행
        prob = model.predict_proba(X)[:, 1][0]
        approved = int(prob >= 0.5)

        # 7. 결과 반환
        return {"approved": approved, "predict_loan": round(prob, 4)}

    except KeyError as e:
        # 요청에 필드가 누락된 경우
        print("=== DEBUG: KeyError traceback ===")
        print(traceback.format_exc())
        raise ValueError(f"Missing field in request: {e}")
    except Exception as e:
        # 그 외 예측 실패
        print("=== DEBUG: Prediction Exception traceback ===")
        print(traceback.format_exc())
        raise RuntimeError(f"Prediction failed: {e}")

