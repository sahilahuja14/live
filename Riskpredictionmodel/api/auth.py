from __future__ import annotations

import os

import jwt
from fastapi import Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer


_BEARER_HEADERS = {"WWW-Authenticate": "Bearer"}
_API_KEY_SCHEME = APIKeyHeader(
    name="x-api-key",
    scheme_name="RiskApiKey",
    description="Optional service-to-service key for protected risk routes.",
    auto_error=False,
)
_DASHBOARD_BEARER_SCHEME = HTTPBearer(
    scheme_name="DashboardBearer",
    description="Dashboard access token issued by /api/auth/login.",
    auto_error=False,
)


def _unauthorized(detail: str = "Authentication required.") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers=_BEARER_HEADERS,
    )


def _expected_api_key() -> str:
    return (os.getenv("API_KEY") or "").strip()


def _dashboard_jwt_secret() -> str:
    return (os.getenv("DASHBOARD_JWT_SECRET") or os.getenv("SECRET_KEY") or "").strip()


def _dashboard_jwt_algorithm() -> str:
    return (os.getenv("DASHBOARD_JWT_ALGORITHM") or "HS256").strip() or "HS256"


def _validate_dashboard_access_token(authorization: str | None) -> dict | None:
    header = str(authorization or "").strip()
    if not header:
        return None

    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _unauthorized("Invalid bearer token.")

    secret = _dashboard_jwt_secret()
    if not secret:
        raise _unauthorized("Dashboard token validation is not configured.")

    try:
        payload = jwt.decode(token.strip(), secret, algorithms=[_dashboard_jwt_algorithm()])
    except jwt.ExpiredSignatureError:
        raise _unauthorized("Token has expired.")
    except jwt.InvalidTokenError:
        raise _unauthorized("Invalid bearer token.")

    username = str(payload.get("sub") or "").strip()
    token_type = str(payload.get("type") or "").strip().lower()
    if not username or token_type != "access":
        raise _unauthorized("Access token required.")
    return payload


def require_api_key(
    x_api_key: str | None = Security(_API_KEY_SCHEME),
    bearer_credentials: HTTPAuthorizationCredentials | None = Security(_DASHBOARD_BEARER_SCHEME),
    authorization: str | None = Header(default=None, include_in_schema=False),
) -> None:
    expected_api_key = _expected_api_key()
    provided_api_key = (x_api_key or "").strip()
    if expected_api_key and provided_api_key == expected_api_key:
        return

    if bearer_credentials:
        _validate_dashboard_access_token(f"{bearer_credentials.scheme} {bearer_credentials.credentials}")
        return

    if authorization:
        _validate_dashboard_access_token(authorization)
        return

    if expected_api_key or _dashboard_jwt_secret():
        raise _unauthorized()

    return


def is_valid_websocket_api_key(api_key: str | None) -> bool:
    expected_api_key = _expected_api_key()
    provided_api_key = str(api_key or "").strip()
    if not expected_api_key:
        return False
    return provided_api_key == expected_api_key
