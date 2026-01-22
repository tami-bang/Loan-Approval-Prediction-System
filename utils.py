# utils.py -> 예외 처리 함수

from fastapi import Request
from fastapi.responses import JSONResponse

def register_exception_handlers(app):
    @app.exception_handler(ValueError)  # 4-(1) 충족
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content={"error": str(exc)}
        )
