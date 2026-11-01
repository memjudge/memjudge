"""GitHub login for the /run form, through the MemJudge GitHub App's user authorisation.

Only the numeric GitHub user id is used, and only as an HMAC (``user_hash``) in run
records: it is never published. After login the API hands the site a short-lived session
token in the URL fragment; the site sends it back as a bearer token. The site and the API
are on different origins, so no cookie is used for the session.

The site starts each login with a random login id, which the OAuth state carries and the
callback hands back with the token. The site takes a token only with the login id it
started, so a link carrying someone else's token cannot sign a visitor in as them.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx
import jwt

GITHUB_AUTHORIZE = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN = "https://github.com/login/oauth/access_token"
GITHUB_USER = "https://api.github.com/user"
SESSION_SECONDS = 8 * 3600
STATE_SECONDS = 600


class LoginError(Exception):
    """Login failed; the message is safe to show."""


@dataclass(frozen=True)
class OAuthApp:
    client_id: str
    client_secret: str

    @classmethod
    def from_json(cls, data: bytes) -> OAuthApp:
        d = json.loads(data)
        return cls(d["client_id"], d["client_secret"])


def authorize_url(app: OAuthApp, redirect_uri: str, state: str) -> str:
    """GitHub's sign-in, with its account picker so a visitor can pick another account."""
    query = urlencode(
        {
            "client_id": app.client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "prompt": "select_account",
        }
    )
    return f"{GITHUB_AUTHORIZE}?{query}"


def github_user_id(http: httpx.Client, app: OAuthApp, code: str, redirect_uri: str) -> str:
    """Exchange the OAuth code and return the user's numeric GitHub id."""
    resp = http.post(
        GITHUB_TOKEN,
        headers={"Accept": "application/json"},
        data={
            "client_id": app.client_id,
            "client_secret": app.client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
        },
    )
    token = resp.json().get("access_token") if resp.status_code == 200 else None
    if not token:
        raise LoginError("GitHub did not accept the login. Try again.")
    user = http.get(
        GITHUB_USER,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    if user.status_code != 200:
        raise LoginError("Could not read your GitHub account. Try again.")
    return str(user.json()["id"])


def _sign(secret: bytes, claims: dict[str, object], seconds: int) -> str:
    now = int(time.time())
    return jwt.encode({**claims, "iat": now, "exp": now + seconds}, secret, "HS256")


def _read(secret: bytes, token: str, purpose: str) -> dict[str, object] | None:
    try:
        claims: dict[str, object] = jwt.decode(token, secret, ["HS256"])
    except jwt.InvalidTokenError:
        return None
    return claims if claims.get("purpose") == purpose else None


def issue_state(secret: bytes, nonce: str, login_id: str) -> str:
    claims: dict[str, object] = {"purpose": "oauth-state", "nonce": nonce, "login_id": login_id}
    return _sign(secret, claims, STATE_SECONDS)


def read_state(secret: bytes, state: str) -> tuple[str, str] | None:
    """The browser nonce and the site's login id that an OAuth state carries."""
    claims = _read(secret, state, "oauth-state")
    if not claims or "login_id" not in claims:
        return None
    return str(claims["nonce"]), str(claims["login_id"])


def issue_session(secret: bytes, user_id: str) -> str:
    return _sign(secret, {"purpose": "session", "sub": user_id}, SESSION_SECONDS)


def session_user(secret: bytes, token: str) -> str | None:
    claims = _read(secret, token, "session")
    return str(claims["sub"]) if claims else None


def user_hash(secret: bytes, user_id: str) -> str:
    return hmac.new(secret, b"user:" + user_id.encode(), hashlib.sha256).hexdigest()[:32]
