"""The Spotify session on disk: one small file, encrypted for the current Windows account (DPAPI).

Nothing in here prints, logs or raises with a token in the text. A file that cannot be read or
decrypted simply means "not signed in".
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import threading
from dataclasses import dataclass

from npass import settings as _settings

_ENTROPY = b"N-Pass spotify session v1"   # ties the blob to this app; not a secret
_UI_FORBIDDEN = 0x1                       # CRYPTPROTECT_UI_FORBIDDEN: never pop up a dialog
_lock = threading.Lock()


@dataclass(frozen=True, repr=False)
class TokenSet:
    access_token: str
    refresh_token: str
    expires_at: float          # time.time() when the access token stops working
    scope: str = ""

    def __repr__(self) -> str:             # never show the contents, anywhere
        return "TokenSet(<hidden>)"

    __str__ = __repr__

    def expired(self, now: float, margin_s: float = 60.0) -> bool:
        return now >= self.expires_at - margin_s


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_uint32), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _blob(data: bytes) -> tuple[_Blob, ctypes.Array]:
    buf = ctypes.create_string_buffer(data, len(data))
    return _Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), buf


def _dpapi(data: bytes, protect: bool) -> bytes:
    """Encrypt or decrypt with the current user's Windows key. Raises OSError (no data in it)."""
    if sys.platform != "win32":
        raise OSError("Windows data protection is not available on this system")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    fn = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    fn.argtypes = [ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.POINTER(_Blob), ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(_Blob)]
    fn.restype = ctypes.c_int
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p

    src, _keep1 = _blob(data)
    ent, _keep2 = _blob(_ENTROPY)
    out = _Blob()
    ok = fn(ctypes.byref(src), None, ctypes.byref(ent), None, None, _UI_FORBIDDEN,
            ctypes.byref(out))
    if not ok:
        raise OSError("Windows could not " + ("protect" if protect else "open")
                      + " the Spotify session file")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(out.pbData, ctypes.c_void_p))


def save_tokens(tokens: TokenSet) -> None:
    """Write the session file atomically (temp file, then swap)."""
    raw = json.dumps({
        "v": 1,
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "expires_at": float(tokens.expires_at),
        "scope": tokens.scope,
    }).encode("utf-8")
    sealed = _dpapi(raw, protect=True)
    path = _settings.token_path()
    tmp = path.with_suffix(".tmp")
    with _lock:
        try:
            with open(tmp, "wb") as f:
                f.write(sealed)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass


def load_tokens() -> TokenSet | None:
    """The saved session, or None when there is none or the file is unreadable / corrupt."""
    try:
        with _lock:
            sealed = _settings.token_path().read_bytes()
        if not sealed:
            return None
        data = json.loads(_dpapi(sealed, protect=False).decode("utf-8"))
        access, refresh = data["access_token"], data["refresh_token"]
        if not isinstance(access, str) or not isinstance(refresh, str) or not refresh:
            return None
        return TokenSet(access, refresh, float(data.get("expires_at", 0.0)),
                        str(data.get("scope", "")))
    except Exception:           # any damage at all = logged out; never echo what was in the file
        return None


def delete_tokens() -> None:
    with _lock:
        for p in (_settings.token_path(), _settings.token_path().with_suffix(".tmp")):
            try:
                p.unlink()
            except OSError:
                pass
