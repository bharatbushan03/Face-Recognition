from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

class AppException(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code

class FaceNotFoundError(AppException):
    def __init__(self, message="No face detected in the image."):
        super().__init__(message=message, status_code=400)

class MultipleFacesError(AppException):
    def __init__(self, message="Multiple faces detected. Please ensure only one face is visible."):
        super().__init__(message=message, status_code=400)

class ImageProcessError(AppException):
    def __init__(self, message="Failed to process image. Ensure it is a valid format."):
        super().__init__(message=message, status_code=400)

async def app_exception_handler(request: Request, exc: AppException):
    logger.warning(
        "AppException on %s %s: %s",
        request.method,
        request.url.path,
        exc.message,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.message,
            "code": exc.status_code
        }
    )

async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled server error on %s %s: %s",
        request.method,
        request.url.path,
        str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error encountered.",
            "code": 500
        }
    )

def setup_exception_handlers(app):
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)
