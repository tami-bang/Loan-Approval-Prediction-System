# app/config.py
from pydantic import BaseSettings

# 환경 변수 기반 설정 클래스 정의
class Settings(BaseSettings):
    # 모델 파일 경로 (필수)
    model_path: str
    # FastAPI 서버 호스트, 기본값 "0.0.0.0"
    host: str = "0.0.0.0"
    # FastAPI 서버 포트, 기본값 8000
    port: int = 8000
    # 로깅 레벨, 기본값 "info"
    log_level: str = "info"

    # Pydantic Config 클래스
    # .env 파일에서 환경 변수 읽기
    class Config:
        env_file = "../.env"

# 설정 인스턴스 생성
# 다른 모듈에서 import하여 사용
settings = Settings()
