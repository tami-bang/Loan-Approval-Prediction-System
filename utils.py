# utils.py

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# 예외 처리 핸들러 등록
def register_exception_handlers(app):

    # ValueError 처리
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"error": str(exc)})

    # 요청 검증 실패 처리
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": "Invalid request", "details": exc.errors()}
        )

    # 런타임 오류 처리
    @app.exception_handler(RuntimeError)
    async def runtime_exception_handler(request: Request, exc: RuntimeError):
        return JSONResponse(status_code=500, content={"error": str(exc)})

