# utils.py
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# FastAPI 앱에 커스텀 예외 핸들러 등록
def register_exception_handlers(app):
    # ValueError 발생 시 400 Bad Request로 응답
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"error": str(exc)})

    # 요청 데이터 검증 실패 시 422 Unprocessable Entity로 응답
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": "Invalid request", "details": exc.errors()}
        )

    # 서버 내부 오류(RuntimeError) 발생 시 500 Internal Server Error로 응답
    @app.exception_handler(RuntimeError)
    async def runtime_exception_handler(request: Request, exc: RuntimeError):
        return JSONResponse(status_code=500, content={"error": str(exc)})
