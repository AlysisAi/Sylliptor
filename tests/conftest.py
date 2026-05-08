from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest

from sylliptor_agent_cli.mcp.oauth import build_pkce_challenge


class OAuthFixtureServer:
    def __init__(self) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _OAuthFixtureHandler)
        self._server.oauth_fixture = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.reset()

    def reset(self) -> None:
        self.authorization_server_path = ""
        self.protected_resource_path = "/protected"
        self.challenge_includes_resource_metadata = True
        self.serve_rfc8414 = True
        self.serve_path_protected_resource_metadata = True
        self.serve_root_protected_resource_metadata = True
        self.serve_oidc_inserted_metadata = True
        self.serve_oidc_appended_metadata = True
        self.protected_resource_payload_override: dict[str, Any] | None = None
        self.path_protected_resource_payload_override: dict[str, Any] | None = None
        self.authorization_metadata_override: dict[str, Any] | None = None
        self.oidc_metadata_override: dict[str, Any] | None = None
        self.request_log: list[str] = []
        self.authorize_requests: list[dict[str, str]] = []
        self.token_requests: list[dict[str, str]] = []
        self.expected_code_challenge: str | None = None
        self.valid_tokens = {"test_access", "refreshed_access"}
        self.issue_refresh_token = True
        self.rotate_refresh_token = True
        self.authorization_code_response_status = HTTPStatus.OK
        self.authorization_code_response_override: dict[str, Any] | None = None
        self.refresh_response_status = HTTPStatus.OK
        self.refresh_response_override: dict[str, Any] | None = None
        self.expected_authorize_resource: str | None = None
        self.expected_token_resource: str | None = None

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    @property
    def protected_url(self) -> str:
        return f"{self.base_url}{self.protected_resource_path}"

    @property
    def resource_metadata_url(self) -> str:
        return f"{self.base_url}/.well-known/oauth-protected-resource"

    @property
    def path_resource_metadata_path(self) -> str:
        return f"/.well-known/oauth-protected-resource{self.protected_resource_path}"

    @property
    def authorization_server_url(self) -> str:
        return f"{self.base_url}{self.authorization_server_path}"

    @property
    def authorization_endpoint(self) -> str:
        return f"{self.base_url}/authorize"

    @property
    def token_endpoint(self) -> str:
        return f"{self.base_url}/token"

    @property
    def rfc8414_metadata_path(self) -> str:
        if self.authorization_server_path:
            return f"/.well-known/oauth-authorization-server{self.authorization_server_path}"
        return "/.well-known/oauth-authorization-server"

    @property
    def oidc_metadata_path(self) -> str:
        return "/.well-known/openid-configuration"

    @property
    def oidc_inserted_metadata_path(self) -> str:
        return f"/.well-known/openid-configuration{self.authorization_server_path}"

    @property
    def oidc_appended_metadata_path(self) -> str:
        return f"{self.authorization_server_path}/.well-known/openid-configuration"

    def protected_resource_payload(self) -> dict[str, Any]:
        if self.protected_resource_payload_override is not None:
            return dict(self.protected_resource_payload_override)
        return {
            "resource": self.protected_url,
            "authorization_servers": [self.authorization_server_url],
        }

    def path_protected_resource_payload(self) -> dict[str, Any]:
        if self.path_protected_resource_payload_override is not None:
            return dict(self.path_protected_resource_payload_override)
        return self.protected_resource_payload()

    def authorization_metadata_payload(self) -> dict[str, Any]:
        if self.authorization_metadata_override is not None:
            return dict(self.authorization_metadata_override)
        return {
            "issuer": self.authorization_server_url,
            "authorization_endpoint": self.authorization_endpoint,
            "token_endpoint": self.token_endpoint,
            "code_challenge_methods_supported": ["S256"],
            "response_types_supported": ["code"],
        }

    def oidc_metadata_payload(self) -> dict[str, Any]:
        if self.oidc_metadata_override is not None:
            return dict(self.oidc_metadata_override)
        return {
            "issuer": self.authorization_server_url,
            "authorization_endpoint": self.authorization_endpoint,
            "token_endpoint": self.token_endpoint,
            "code_challenge_methods_supported": ["S256"],
            "response_types_supported": ["code"],
        }


class _OAuthFixtureHandler(BaseHTTPRequestHandler):
    server_version = "OAuthFixture/1.0"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    @property
    def fixture(self) -> OAuthFixtureServer:
        return self.server.oauth_fixture  # type: ignore[attr-defined]

    def _json(
        self,
        payload: dict[str, Any],
        *,
        status: int = HTTPStatus.OK,
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        self.fixture.request_log.append(parsed.path)
        if (
            self.fixture.protected_resource_path != "/"
            and parsed.path == self.fixture.path_resource_metadata_path
        ):
            if not self.fixture.serve_path_protected_resource_metadata:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._json(self.fixture.path_protected_resource_payload())
            return
        if parsed.path == "/.well-known/oauth-protected-resource":
            if not self.fixture.serve_root_protected_resource_metadata:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._json(self.fixture.protected_resource_payload())
            return
        if parsed.path == self.fixture.rfc8414_metadata_path:
            if not self.fixture.serve_rfc8414:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._json(self.fixture.authorization_metadata_payload())
            return
        if (
            self.fixture.authorization_server_path
            and self.fixture.authorization_server_path != "/"
            and parsed.path == self.fixture.oidc_inserted_metadata_path
        ):
            if not self.fixture.serve_oidc_inserted_metadata:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._json(self.fixture.oidc_metadata_payload())
            return
        if (
            self.fixture.authorization_server_path
            and self.fixture.authorization_server_path != "/"
            and parsed.path == self.fixture.oidc_appended_metadata_path
        ):
            if not self.fixture.serve_oidc_appended_metadata:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._json(self.fixture.oidc_metadata_payload())
            return
        if parsed.path == self.fixture.oidc_metadata_path:
            self._json(self.fixture.oidc_metadata_payload())
            return
        if parsed.path == "/authorize":
            query = parse_qs(parsed.query)
            normalized_query = {key: values[0] for key, values in query.items() if values}
            self.fixture.authorize_requests.append(normalized_query)
            if query.get("code_challenge_method", [""])[0] != "S256":
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            if query.get("response_type", [""])[0] != "code":
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            redirect_uri = query.get("redirect_uri", [""])[0]
            if not redirect_uri:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            if self.fixture.expected_authorize_resource is not None:
                if query.get("resource", [""])[0] != self.fixture.expected_authorize_resource:
                    self.send_error(HTTPStatus.BAD_REQUEST)
                    return
            self.fixture.expected_code_challenge = query.get("code_challenge", [None])[0]
            state = query.get("state", [""])[0]
            separator = "&" if "?" in redirect_uri else "?"
            self._redirect(
                f"{redirect_uri}{separator}{urlencode({'code': 'TEST_CODE', 'state': state})}"
            )
            return
        if parsed.path == self.fixture.protected_resource_path:
            auth_header = self.headers.get("Authorization")
            if auth_header not in {f"Bearer {token}" for token in self.fixture.valid_tokens}:
                headers: dict[str, str] = {}
                if self.fixture.challenge_includes_resource_metadata:
                    headers["WWW-Authenticate"] = (
                        f'Bearer resource_metadata="{self.fixture.resource_metadata_url}"'
                    )
                else:
                    headers["WWW-Authenticate"] = "Bearer"
                self._json(
                    {"error": "unauthorized"}, status=HTTPStatus.UNAUTHORIZED, headers=headers
                )
                return
            self._json({"ok": True})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        self.fixture.request_log.append(parsed.path)
        if parsed.path != "/token":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        payload = parse_qs(self.rfile.read(length).decode("utf-8"))
        normalized_payload = {key: values[0] for key, values in payload.items() if values}
        self.fixture.token_requests.append(normalized_payload)
        grant_type = payload.get("grant_type", [""])[0]
        if self.fixture.expected_token_resource is not None:
            if payload.get("resource", [""])[0] != self.fixture.expected_token_resource:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
        if grant_type == "authorization_code":
            verifier = payload.get("code_verifier", [""])[0]
            if payload.get("code", [""])[0] != "TEST_CODE":
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            if self.fixture.expected_code_challenge is None:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            if build_pkce_challenge(verifier) != self.fixture.expected_code_challenge:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            response = (
                dict(self.fixture.authorization_code_response_override)
                if self.fixture.authorization_code_response_override is not None
                else {
                    "access_token": "test_access",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                }
            )
            if (
                self.fixture.authorization_code_response_override is None
                and self.fixture.issue_refresh_token
            ):
                response["refresh_token"] = "test_refresh"
            self._json(response, status=int(self.fixture.authorization_code_response_status))
            return
        if grant_type == "refresh_token":
            response = (
                dict(self.fixture.refresh_response_override)
                if self.fixture.refresh_response_override is not None
                else {
                    "access_token": "refreshed_access",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                }
            )
            if self.fixture.refresh_response_override is None and self.fixture.rotate_refresh_token:
                response["refresh_token"] = "rotated_refresh"
            self._json(response, status=int(self.fixture.refresh_response_status))
            return
        self.send_error(HTTPStatus.BAD_REQUEST)


@pytest.fixture
def oauth_fixture_server(monkeypatch: pytest.MonkeyPatch) -> Any:
    server = OAuthFixtureServer()

    async def fixture_safe_http_request(
        method: str,
        url: str,
        *,
        timeout: float = 30.0,
        max_bytes: int = 10 * 1024 * 1024,
        allow_redirects: bool = True,
        max_redirects: int = 5,
        headers: dict[str, str] | None = None,
        json: Any = None,
        content: bytes | None = None,
    ) -> httpx.Response:
        del max_redirects
        async with httpx.AsyncClient(
            follow_redirects=allow_redirects,
            timeout=httpx.Timeout(timeout),
        ) as client:
            response = await client.request(
                method,
                url,
                headers=headers,
                json=json,
                content=content,
            )
        if len(response.content) > max_bytes:
            from sylliptor_agent_cli.safety import SafeHttpError

            raise SafeHttpError(f"Response body exceeded max_bytes={max_bytes}.")
        return response

    monkeypatch.setattr(
        "sylliptor_agent_cli.mcp.oauth.safe_http_request",
        fixture_safe_http_request,
    )
    try:
        yield server
    finally:
        server.close()
