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
    # 모델 파일이 존재하지 않으면 오류 발생
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")
except Exception as e:
    # 그 외 모델 로드 실패 시 오류 발생
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

# 학습/테스트 CSV 경로
TRAIN_CSV = os.path.join(os.path.dirname(__file__), "data", "loan_train.csv")
TEST_CSV = os.path.join(os.path.dirname(__file__), "data", "loan_test.csv")

# 데이터 통합
loan_data = pd.concat([pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)], ignore_index=True)
# Loan_ID를 인덱스로 설정
loan_data.set_index("Loan_ID", inplace=True)


def predict_loan(features: dict) -> dict:
    """
    features dict 기반으로 승인 여부와 승인 확률 반환
    """
    try:
        # 1. FEATURE_COLUMNS 순서대로 데이터 추출, 누락 컬럼은 기본값 처리
        row = {}
        for col in FEATURE_COLUMNS:
            val = features.get(col, 0 if col in NUMERIC_COLUMNS else "")
            # NaN 체크
            if val is None or (isinstance(val, float) and pd.isna(val)):
                val = 0 if col in NUMERIC_COLUMNS else ""
            row[col] = val

        # 2. 수치형 컬럼 float로 변환
        for col in NUMERIC_COLUMNS:
            try:
                row[col] = float(row[col])
            except Exception:
                row[col] = 0.0  # 변환 실패 시 기본값

        # 3. 범주형 컬럼 str로 변환
        for col in CATEGORICAL_COLUMNS:
            row[col] = str(row[col])

        # 4. DataFrame 생성
        X = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        # 5. 디버깅 출력: DataFrame 내용과 컬럼별 dtype
        print("DEBUG: DataFrame for prediction")
        print(X)
        print("DEBUG: dtypes")
        print(X.dtypes)

        # 6. 예측 수행
        prob = model.predict_proba(X)[:, 1][0]  # 승인 확률
        approved = int(prob >= 0.5)           # 승인 여부

        # 7. 결과 반환
        return {"approved": approved, "predict_loan": round(prob, 4)}

    except KeyError as e:
        # 요청에 필드가 누락된 경우
        print("DEBUG: KeyError traceback")
        print(traceback.format_exc())
        raise ValueError(f"Missing field in request: {e}")
    except Exception as e:
        # 그 외 예측 실패
        print("DEBUG: Prediction Exception traceback")
        print(traceback.format_exc())
        raise RuntimeError(f"Prediction failed: {e}")


def predict_loan_by_id(loan_id: str) -> dict:
    """
    Loan_ID 기반으로 승인 여부, 승인 확률, 개인정보 반환
    """
    # 주어진 loan_id가 데이터프레임의 인덱스에 존재하지 않으면 오류 발생
    if loan_id not in loan_data.index:
        raise ValueError(f"Loan_ID '{loan_id}' not found")

    # loan_id에 해당하는 행(row) 가져오기
    row = loan_data.loc[loan_id]

    # 안전한 float/int 변환 + NaN 처리
    def safe_float(val):
        try:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return 0.0
            return float(val)
        except Exception:
            return 0.0

    def safe_int(val):
        try:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return 0
            return int(val)
        except Exception:
            return 0

    # 개별 컬럼 값 추출 및 타입 변환
    applicant_income = safe_float(row.get("ApplicantIncome"))  # 신청인 소득
    coapplicant_income = safe_float(row.get("CoapplicantIncome"))  # 공동 신청인 소득
    dependents = safe_int(row.get("Dependents"))  # 부양가족 수
    loan_amount = safe_float(row.get("LoanAmount"))  # 대출금액
    loan_term = safe_float(row.get("Loan_Amount_Term"))  # 대출 기간
    credit_history = safe_int(row.get("Credit_History"))  # 신용 이력
    total_income = applicant_income + coapplicant_income

    # 모델 입력용 feature 딕셔너리 생성
    features = {
        "Gender": str(row.get("Gender", "")),  # 성별 정보
        "Married": str(row.get("Married", "")),  # 결혼 여부
        "Dependents": dependents,  # 부양가족 수
        "Education": str(row.get("Education", "")),  # 학력 정보
        "Self_Employed": str(row.get("Self_Employed", "")),  # 자영업 여부
        "Property_Area": str(row.get("Property_Area", "")),  # 거주지역
        "LoanAmount": loan_amount,  # 대출금액
        "Loan_Amount_Term": loan_term,  # 대출 기간
        "TotalIncome": total_income,  # 총 소득
        "Credit_History": credit_history,  # 신용 이력
        "IncomePerDependent": total_income / (dependents + 1),  # 부양가족 1인당 소득
        "LoanAmountToIncome": loan_amount / total_income if total_income != 0 else 0,  # 대출금 대비 소득 비율
        "Education_SelfEmployed": f"{row.get('Education', '')}_{row.get('Self_Employed', '')}"  # 학력과 자영업 여부 결합
    }

    # features를 기반으로 대출 승인 여부 예측 후 결과 반환
    prediction = predict_loan(features)

    # 개인정보 컬럼 안전 처리
    personal_info_cols = [
        "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
        "Loan_Amount_Term", "Credit_History", "Property_Area"
    ]

    # NaN, None 처리 후 문자열/숫자로 변환
    personal_info = {}
    for col in personal_info_cols:
        val = row.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            val = "미입력"
        elif isinstance(val, pd.Timestamp):
            val = str(val)
        elif isinstance(val, (int, float)):
            val = val
        else:
            val = str(val)
        personal_info[col] = val

    # 최종 반환값
    result = {
        "approved": prediction["approved"],  # 승인 여부
        "predict_loan": prediction["predict_loan"],  # 승인 확률
        "personal_info": personal_info  # 개인정보 포함
    }

    # 최종 결과 반환 (예측 + 개인정보)
    return result

