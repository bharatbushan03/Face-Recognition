from __future__ import annotations

import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials


http_basic = HTTPBasic(auto_error=False)


def is_admin_auth_enabled() -> bool:
    username = os.getenv("FR_ADMIN_USERNAME", "").strip()
    password = os.getenv("FR_ADMIN_PASSWORD", "").strip()
    return bool(username and password)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Admin authentication required",
        headers={"WWW-Authenticate": "Basic"},
    )


def require_admin_auth(credentials: HTTPBasicCredentials | None = Depends(http_basic)) -> None:
    """Protect sensitive endpoints when FR_ADMIN_USERNAME/PASSWORD are configured."""
    if not is_admin_auth_enabled():
        return

    if credentials is None:
        raise _unauthorized()

    expected_username = os.getenv("FR_ADMIN_USERNAME", "")
    expected_password = os.getenv("FR_ADMIN_PASSWORD", "")

    username_ok = secrets.compare_digest(credentials.username, expected_username)
    password_ok = secrets.compare_digest(credentials.password, expected_password)

    if not (username_ok and password_ok):
        raise _unauthorized()
