# app/utils.py
import logging
import time
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR

# Logger 설정
# API 요청/응답 및 예외 로그 기록용
logger = logging.getLogger("loan_api")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# 콘솔 출력용 핸들러 설정
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# 헬스 체크 함수
# 서버 상태 확인용 기본 응답, 필요 시 DB/모델 상태 추가 가능
def health_check():
    return {"status": "ok"}


# 요청 처리 시간 로깅 미들웨어
# 각 요청의 처리 시간 측정 후 로그 기록
async def log_request_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"{request.method} {request.url.path} completed in {elapsed_ms:.2f}ms")
    return response


# RequestValidationError 처리 핸들러
# 요청 데이터 검증 실패 시 400 응답 및 로그 기록
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={"detail": exc.errors()},
    )


# ValueError 처리 핸들러
# 값 오류 발생 시 400 응답 및 에러 메시지 로그 기록
async def value_error_handler(request: Request, exc: ValueError):
    logger.error(f"ValueError: {str(exc)}")
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={"error": str(exc)},
    )


# Generic Exception 처리 핸들러
# 예상치 못한 서버 오류 발생 시 500 응답 및 스택트레이스 로그 기록
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"},
    )


# 예외 핸들러 등록 유틸
# FastAPI 앱에 커스텀 예외 핸들러 연결
def register_exception_handlers(app):
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValueError, value_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

