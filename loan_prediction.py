# loan_prediction.py -> 모델 학습 완료 (이미 joblib 저장)
# 이 파일은 대출 신청자의 데이터를 보고 '승인될 확률'을 예측하는 AI 모델을 만드는 코드입니다.
# 여기서 목표는:
#   - 대출 신청자가 승인될 확률을 계산 (0~1)
#   - 모델이 학습할 수 있는 형태로 데이터 전처리
#   - 학습된 모델로 예측하고 평가하기

# Kaggle Loan Prediction Problem Dataset
# 링크: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
import os
import pandas as pd                  # 엑셀처럼 데이터를 표 형태로 다루기 위해 필요
import numpy as np                   # 숫자 계산, 수학 함수 사용
import matplotlib.pyplot as plt      # 데이터 시각화 (그래프)
import seaborn as sns                # 통계적 시각화, 히트맵 등
import joblib                        # API 

# 머신러닝용 도구
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
# 데이터 불균형 처리
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# 공통 상수
# 공통 상수
MODEL_PATH = "models/loan_logreg_pipeline_enhanced.joblib"


# ===============================
# 모델 저장 함수
# ===============================
def save_model(model, model_path: str):
    """
    학습 완료 된 모델 저장

    Args:
        model: 학습 완료 된 모델 (Pipeline 포함)
        model_path (str): 저장할 파일 이름(경로 포함)
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[OK] Saved model -> {model_path}")

# 1. 데이터 불러오기
# CSV 파일에서 데이터를 읽어옵니다.
# train_df: 학습용 데이터, test_df: 실제 예측용 데이터
train_df = pd.read_csv("data/loan_train.csv")
test_df = pd.read_csv("data/loan_test.csv")

# 데이터 구조 확인 (몇 행, 몇 열)
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


# 2. Pandas 전처리 5단계

# 2-1. 타입 변환
# Dependents 컬럼에 '3+'라는 값이 있어, 계산하려면 숫자로 바꿔야 합니다.
train_df['Dependents'] = train_df['Dependents'].replace('3+', 3)
test_df['Dependents'] = test_df['Dependents'].replace('3+', 3)

# 2-2. 결측치(NA 값) 처리
# 어떤 열은 비어있거나 NA가 있음
# 숫자형은 중앙값(median), 범주형(문자)는 최빈값(mode)으로 채우는게 안정적
num_cols = ['LoanAmount','Loan_Amount_Term','ApplicantIncome','CoapplicantIncome','Credit_History']
cat_cols = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']

def fill_missing(df, num_cols, cat_cols, median_map=None, mode_map=None):
    """
    - df: 데이터프레임
    - num_cols: 숫자 컬럼 이름 리스트
    - cat_cols: 범주 컬럼 이름 리스트
    - median_map/mode_map: 이전에 계산한 값 재사용
    """
    if median_map is None:
        median_map = {col: df[col].median() for col in num_cols}  # 숫자 중앙값 계산
    if mode_map is None:
        mode_map = {col: df[col].mode()[0] for col in cat_cols}   # 문자 중앙값 계산
    for col in num_cols:
        df[col] = df[col].fillna(median_map[col])
    for col in cat_cols:
        df[col] = df[col].fillna(mode_map[col])
    return df, median_map, mode_map

# 학습용 데이터 채우기
train_df, median_map, mode_map = fill_missing(train_df, num_cols, cat_cols)
# 테스트 데이터는 학습 데이터 기준 값으로 채움 (모델 학습과 일관성 유지)
test_df, _, _ = fill_missing(test_df, num_cols, cat_cols, median_map, mode_map)

# 2-3. 이상치 처리 (IQR)
# 너무 큰 값/작은 값은 모델에 나쁜 영향을 줌
# IQR(사분위 범위) 기준으로 클리핑
def clip_iqr(df, cols):
    df_copy = df.copy()
    for col in cols:
        Q1 = df_copy[col].quantile(0.25)  # 하위 25%
        Q3 = df_copy[col].quantile(0.75)  # 상위 75%
        IQR = Q3 - Q1
        # 1.5 * IQR 범위를 넘어가는 값은 잘라줌
        df_copy[col] = np.clip(df_copy[col], Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    return df_copy

# 적용
train_df = clip_iqr(train_df, ['LoanAmount','ApplicantIncome','CoapplicantIncome'])
test_df = clip_iqr(test_df, ['LoanAmount','ApplicantIncome','CoapplicantIncome'])

# 2-4. 컬럼 선택
# 모델에 넣을 특징(feature)만 선택
selected_features = ['Gender','Married','Dependents','Education','Self_Employed',
                     'Property_Area','LoanAmount','Loan_Amount_Term','ApplicantIncome','CoapplicantIncome','Credit_History']
train_df = train_df[selected_features + ['Loan_Status']]  # 목표값 포함
test_df = test_df[selected_features]                      # 테스트에는 목표값 없음

# 2-5. 파생 변수 생성
# 모델 성능 향상을 위해 간단한 계산으로 새 변수 추가
train_df['TotalIncome'] = train_df['ApplicantIncome'] + train_df['CoapplicantIncome']
test_df['TotalIncome'] = test_df['ApplicantIncome'] + test_df['CoapplicantIncome']

train_df['Dependents'] = train_df['Dependents'].astype(int)
test_df['Dependents'] = test_df['Dependents'].astype(int)

train_df['IncomePerDependent'] = train_df['TotalIncome'] / (train_df['Dependents'] + 1)
test_df['IncomePerDependent'] = test_df['TotalIncome'] / (test_df['Dependents'] + 1)

train_df['LoanAmountToIncome'] = train_df['LoanAmount'] / train_df['TotalIncome']
test_df['LoanAmountToIncome'] = test_df['LoanAmount'] / test_df['TotalIncome']

# Education과 Self_Employed 합쳐서 새로운 범주 생성
train_df['Education_SelfEmployed'] = train_df['Education'] + "_" + train_df['Self_Employed']
test_df['Education_SelfEmployed'] = test_df['Education'] + "_" + test_df['Self_Employed']

# 3. Feature / Target 정의
# 입력(X)과 목표값(y) 정의
feature_cols = ['Gender','Married','Dependents','Education','Self_Employed',
                'Property_Area','LoanAmount','Loan_Amount_Term','TotalIncome',
                'Credit_History','IncomePerDependent','LoanAmountToIncome',
                'Education_SelfEmployed']

X = train_df[feature_cols]                        # 모델 입력값
y = train_df['Loan_Status'].map({'N':0, 'Y':1})   # 목표값 0(N), 1(Y)로 변환

# 4. Train/Validation 분리
# 학습용/검증용 데이터 나누기
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# stratify=y → 승인/거절 비율이 학습/검증 둘 다 비슷하게 유지

# 5. 전처리 Pipeline 정의
# 수치형 데이터는 로그 변환 후 표준화, 범주형 데이터는 원-핫 인코딩
log_transformer = FunctionTransformer(func=np.log1p)

num_features = ['LoanAmount','Loan_Amount_Term','TotalIncome','Credit_History','Dependents','IncomePerDependent','LoanAmountToIncome']
cat_features = ['Gender','Married','Education','Self_Employed','Property_Area','Education_SelfEmployed']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('log', log_transformer), ('scaler', StandardScaler())]), num_features),  # 수치형
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)                               # 범주형
    ]
)

# 6. Logistic Regression 모델 정의
logreg = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', C=1.0, random_state=42)

# 전처리 → SMOTE → Logistic Regression 전체 pipeline
pipe = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', logreg)
])


# 7. 모델 학습
pipe.fit(X_train, y_train)              # 학습
y_pred = pipe.predict(X_val)            # 검증용 예측 (0/1)
y_prob = pipe.predict_proba(X_val)[:,1] # 승인 확률(0~1)

# 8. 모델 평가
acc = accuracy_score(y_val, y_pred)           # 맞춘 비율
auc = roc_auc_score(y_val, y_prob)            # 확률 기반 성능
cm = confusion_matrix(y_val, y_pred)          # 실제 vs 예측 비교
report = classification_report(y_val, y_pred) # 정밀도, 재현율, F1

print(f"\n=== Logistic Regression (Enhanced) ===")
print(f"Accuracy (정확도): {acc:.4f}")
print(f"AUC (ROC): {auc:.4f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)

# 9. 예측 확률 분포 시각화
plt.figure(figsize=(10,6))
sns.histplot(y_prob, bins=20, kde=True, color='blue', alpha=0.6)
plt.title("Predicted Approval Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.show()

# 10. Feature Importance 분석
# Logistic Regression 계수로 중요 feature 확인
feature_names = list(pipe.named_steps['preprocessor'].named_transformers_['num'].feature_names_in_) + \
                list(pipe.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out())

coefs = pipe.named_steps['classifier'].coef_[0]
feat_importance = pd.DataFrame({'feature': feature_names, 'coef': coefs})
feat_importance['abs_coef'] = feat_importance['coef'].abs()
feat_importance.sort_values('abs_coef', ascending=False, inplace=True)

# Top 15 중요 feature 시각화
plt.figure(figsize=(10,6))
sns.barplot(
    data=feat_importance.head(15),
    x='abs_coef',
    y='feature',
    hue='feature',       # y값을 hue로 지정
    palette='viridis',
    dodge=False,         # 막대 겹치지 않도록 설정
    legend=False         # 범례 제거
)
plt.title("Top 15 Feature Importance (Logistic Regression)")
plt.show()

# 11. 전처리된 CSV 저장
test_df_processed = pd.DataFrame(
    preprocessor.transform(test_df),
    columns=num_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out())
)
test_df_processed.to_csv("data/loan_test_preprocessed_enhanced.csv", index=False)
train_df.to_csv("data/loan_train_preprocessed_enhanced.csv", index=False)
print("전처리된 CSV 저장 완료")

# 12. 모델 저장
save_model(
    model=pipe,
    model_path=MODEL_PATH
)
