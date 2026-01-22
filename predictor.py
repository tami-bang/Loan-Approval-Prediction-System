# predictor.py -> 모델 학습 완료 (이미 joblib 저장)

import joblib
import pandas as pd
import os

MODEL_PATH = "models/loan_logreg_pipeline_enhanced.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")  # 1-(3) 충족

model = joblib.load(MODEL_PATH)  # 1-(3) 충족

def predict_loan(data: dict):
    """입력 데이터를 받아서 승인 여부와 확률을 반환"""
    try:
        df = pd.DataFrame([data])
        prob = model.predict_proba(df)[:,1][0]  # 승인 확률 계산 → 3-(2,3)
        approved = int(prob >= 0.5)            # 승인 여부 결정 → 3-(2)
        return {"approved": approved, "predict_loan": float(prob)}
    except Exception as e:
        raise ValueError(f"예측 처리 중 오류 발생: {e}")  # 4-(1) 충족
