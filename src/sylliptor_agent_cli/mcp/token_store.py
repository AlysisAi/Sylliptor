from __future__ import annotations

import base64
import ctypes
import getpass
import json
import logging
import os
import platform
import secrets
import tempfile
from contextlib import suppress
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from ..atomic_io import _fsync_dir
from .errors import (
    McpTokenStoreCorruptError,
    McpTokenStoreError,
    McpTokenStoreMigrationError,
    McpTokenStoreUnavailableError,
    McpTokenStoreVersionError,
)

CURRENT_ENVELOPE_VERSION = 2
TOKEN_STORE_AAD = b"sylliptor-mcp-oauth-store"
KEY_SOURCE_KEYRING = "keyring"
KEY_SOURCE_WEAK_FALLBACK = "weak-derived-fallback"
KEY_SOURCE_DPAPI = "dpapi"
KEY_SOURCE_FILESYSTEM = "filesystem-random"

_ENVELOPE_KEYS = frozenset({"version", "key_source", "nonce", "ciphertext"})
_KEYRING_SERVICE = "sylliptor-agent-cli"
_KEYRING_ACCOUNT = "mcp-oauth-token-store-master-key"
_WEAK_SALT_FILE_NAME = "mcp_oauth_tokens.salt"
_DPAPI_KEY_FILE_NAME = "mcp_oauth_tokens.dpapi"
_AES_KEY_BYTES = 32
_AES_GCM_NONCE_BYTES = 12
_CRYPTPROTECT_UI_FORBIDDEN = 0x1

logger = logging.getLogger(__name__)
# Library loggers must not fall through to logging.lastResort, which writes
# WARNING records directly to stderr and corrupts prompt_toolkit full-screen
# applications. Configured application/test handlers still receive records via
# normal propagation.
logger.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class _KeyMaterial:
    source: str
    key: bytes


@dataclass(frozen=True)
class TokenPayloadLoadResult:
    payload: dict[str, Any]
    legacy_plaintext: bool = False


def load_token_payload(path: Path) -> dict[str, Any]:
    return load_token_payload_result(path).payload


def load_token_payload_result(path: Path) -> TokenPayloadLoadResult:
    if not path.exists():
        return TokenPayloadLoadResult({})
    raw_payload = _read_json_file(path)
    classification = _classify_store_payload(raw_payload)
    if classification == "envelope":
        payload = _decrypt_envelope(path, raw_payload)
        return TokenPayloadLoadResult(payload)
    if classification == "invalid":
        raise McpTokenStoreCorruptError(f"Invalid OAuth token store format: {path}")
    return TokenPayloadLoadResult(raw_payload, legacy_plaintext=True)


def migrate_legacy_token_payload(path: Path, payload: dict[str, Any]) -> None:
    try:
        _write_encrypted_payload(path, payload)
    except Exception as exc:
        raise McpTokenStoreMigrationError(
            f"Failed to migrate legacy plaintext OAuth token store: {path}"
        ) from exc


def save_token_payload(path: Path, payload: dict[str, Any]) -> None:
    _write_encrypted_payload(path, payload)


def weak_fallback_salt_path(store_path: Path) -> Path:
    return store_path.with_name(_WEAK_SALT_FILE_NAME)


def dpapi_master_key_path(store_path: Path) -> Path:
    return store_path.with_name(_DPAPI_KEY_FILE_NAME)


def filesystem_master_key_path(store_path: Path) -> Path:
    """Return the per-store random master-key path used without an OS keyring."""

    return store_path.with_suffix(".key")


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise McpTokenStoreCorruptError(f"Malformed OAuth token store JSON: {path}") from exc
    except OSError as exc:
        raise McpTokenStoreUnavailableError(f"Failed to read OAuth token store: {path}") from exc


def _classify_store_payload(payload: object) -> Literal["envelope", "legacy", "invalid"]:
    if not isinstance(payload, dict):
        return "invalid"
    keys = frozenset(payload)
    marker_keys = keys & _ENVELOPE_KEYS
    if not marker_keys:
        return "legacy"
    marker_values = tuple(payload[key] for key in marker_keys)
    if all(_looks_like_legacy_record_payload(value) for value in marker_values):
        return "legacy"
    if all(_looks_like_envelope_marker_value(key, payload[key]) for key in marker_keys):
        return "envelope"
    return "invalid"


def _looks_like_legacy_record_payload(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return all(
        isinstance(value.get(key), str) for key in ("access_token", "token_type", "expires_at")
    )


def _looks_like_envelope_marker_value(key: str, value: object) -> bool:
    if key == "version":
        return isinstance(value, int)
    return isinstance(value, str)


def _validate_envelope(payload: dict[str, Any], *, path: Path) -> tuple[int, str, bytes, bytes]:
    if frozenset(payload) != _ENVELOPE_KEYS:
        raise McpTokenStoreCorruptError(f"Malformed OAuth token store envelope: {path}")
    version = payload.get("version")
    if not isinstance(version, int):
        raise McpTokenStoreCorruptError(f"OAuth token store envelope version is invalid: {path}")
    if version > CURRENT_ENVELOPE_VERSION:
        raise McpTokenStoreVersionError(
            f"OAuth token store envelope version {version} is newer than supported "
            f"version {CURRENT_ENVELOPE_VERSION}: {path}"
        )
    if version < 1:
        raise McpTokenStoreCorruptError(f"OAuth token store envelope version is invalid: {path}")
    key_source = payload.get("key_source")
    if key_source not in _allowed_key_sources(version):
        raise McpTokenStoreCorruptError(f"OAuth token store key source is invalid: {path}")
    try:
        nonce = base64.b64decode(_require_string(payload.get("nonce")), validate=True)
        ciphertext = base64.b64decode(_require_string(payload.get("ciphertext")), validate=True)
    except (ValueError, TypeError) as exc:
        raise McpTokenStoreCorruptError(
            f"OAuth token store envelope encoding is invalid: {path}"
        ) from exc
    if len(nonce) != _AES_GCM_NONCE_BYTES:
        raise McpTokenStoreCorruptError(f"OAuth token store envelope nonce is invalid: {path}")
    return version, key_source, nonce, ciphertext


def _require_string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("expected non-empty string")
    return value


def _allowed_key_sources(version: int) -> frozenset[str]:
    if version == 1:
        return frozenset({KEY_SOURCE_KEYRING, KEY_SOURCE_WEAK_FALLBACK, KEY_SOURCE_DPAPI})
    return frozenset({KEY_SOURCE_KEYRING, KEY_SOURCE_FILESYSTEM, KEY_SOURCE_DPAPI})


def _decrypt_envelope(path: Path, envelope: dict[str, Any]) -> dict[str, Any]:
    version, stored_source, nonce, ciphertext = _validate_envelope(envelope, path=path)
    key_material = _key_material_for_source(stored_source, path=path)
    try:
        plaintext = AESGCM(key_material.key).decrypt(nonce, ciphertext, TOKEN_STORE_AAD)
    except InvalidTag as exc:
        raise McpTokenStoreCorruptError(f"OAuth token store authentication failed: {path}") from exc
    except ValueError as exc:
        raise McpTokenStoreCorruptError(f"OAuth token store decryption failed: {path}") from exc
    try:
        payload = json.loads(plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise McpTokenStoreCorruptError(
            f"OAuth token store decrypted payload is invalid: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise McpTokenStoreCorruptError(f"OAuth token store decrypted payload is invalid: {path}")
    _maybe_rewrite_envelope_after_read(
        path,
        payload,
        version=version,
        stored_source=stored_source,
    )
    return payload


def _maybe_rewrite_envelope_after_read(
    path: Path,
    payload: dict[str, Any],
    *,
    version: int,
    stored_source: str,
) -> None:
    if version >= CURRENT_ENVELOPE_VERSION and stored_source == KEY_SOURCE_KEYRING:
        return
    try:
        preferred = _preferred_key_material(path)
    except (McpTokenStoreError, OSError) as exc:
        logger.warning(
            "Skipped OAuth credential store rotation after successful decrypt: "
            "path=%s version=%s key_source=%s error_type=%s",
            path,
            version,
            stored_source,
            exc.__class__.__name__,
        )
        return
    if version >= CURRENT_ENVELOPE_VERSION and preferred.source == stored_source:
        return
    try:
        _write_encrypted_payload(path, payload, key_material=preferred)
    except (McpTokenStoreError, OSError) as exc:
        logger.warning(
            "Skipped OAuth credential store rewrite after successful decrypt: "
            "path=%s version=%s key_source=%s preferred_key_source=%s error_type=%s",
            path,
            version,
            stored_source,
            preferred.source,
            exc.__class__.__name__,
        )


def _write_encrypted_payload(
    path: Path,
    payload: dict[str, Any],
    *,
    key_material: _KeyMaterial | None = None,
) -> None:
    key_material = key_material or _preferred_key_material(path)
    if key_material.source not in _allowed_key_sources(CURRENT_ENVELOPE_VERSION):
        raise McpTokenStoreUnavailableError(
            "Refusing to write an OAuth credential envelope with a legacy key source."
        )
    plaintext = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    nonce = secrets.token_bytes(_AES_GCM_NONCE_BYTES)
    ciphertext = AESGCM(key_material.key).encrypt(nonce, plaintext, TOKEN_STORE_AAD)
    envelope = {
        "version": CURRENT_ENVELOPE_VERSION,
        "key_source": key_material.source,
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }
    encoded = (
        json.dumps(envelope, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    try:
        _secure_atomic_write_bytes(path, encoded)
    except OSError as exc:
        raise McpTokenStoreUnavailableError(f"Failed to write OAuth token store: {path}") from exc
    logger.info(
        "Wrote encrypted OAuth credential store: path=%s key_source=%s",
        path,
        key_material.source,
    )


def _preferred_key_material(path: Path) -> _KeyMaterial:
    keyring_material = _try_keyring_material(required=False)
    if keyring_material is not None:
        return keyring_material
    if _platform_system().lower() == "windows":
        return _dpapi_key_material(path)
    return _filesystem_key_material(path)


def _key_material_for_source(source: str, *, path: Path) -> _KeyMaterial:
    if source == KEY_SOURCE_KEYRING:
        material = _try_keyring_material(required=True)
        if material is None:
            raise McpTokenStoreUnavailableError(
                "OS keyring is unavailable for MCP OAuth token store."
            )
        return material
    if source == KEY_SOURCE_DPAPI:
        return _dpapi_key_material(path)
    if source == KEY_SOURCE_FILESYSTEM:
        return _filesystem_key_material(path)
    if source == KEY_SOURCE_WEAK_FALLBACK:
        return _weak_fallback_key_material(path)
    raise McpTokenStoreCorruptError(f"Unsupported OAuth token store key source: {source}")


def _try_keyring_material(*, required: bool) -> _KeyMaterial | None:
    try:
        encoded = _get_keyring_password(_KEYRING_SERVICE, _KEYRING_ACCOUNT)
    except Exception as exc:  # noqa: BLE001
        if required:
            raise McpTokenStoreUnavailableError(
                "Failed to read MCP OAuth key from OS keyring."
            ) from exc
        logger.debug(
            "OS keyring read failed for MCP OAuth token store; falling back.", exc_info=exc
        )
        return None
    if encoded:
        try:
            key = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise McpTokenStoreUnavailableError("MCP OAuth keyring entry is invalid.") from exc
        if len(key) != _AES_KEY_BYTES:
            raise McpTokenStoreUnavailableError("MCP OAuth keyring entry has invalid length.")
        return _KeyMaterial(KEY_SOURCE_KEYRING, key)
    if required:
        raise McpTokenStoreUnavailableError("MCP OAuth keyring entry is missing.")
    key = secrets.token_bytes(_AES_KEY_BYTES)
    try:
        _set_keyring_password(
            _KEYRING_SERVICE,
            _KEYRING_ACCOUNT,
            base64.b64encode(key).decode("ascii"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "OS keyring write failed for MCP OAuth token store; falling back.", exc_info=exc
        )
        return None
    return _KeyMaterial(KEY_SOURCE_KEYRING, key)


def _get_keyring_password(service: str, account: str) -> str | None:
    import keyring

    return keyring.get_password(service, account)


def _set_keyring_password(service: str, account: str, password: str) -> None:
    import keyring

    keyring.set_password(service, account, password)


def _weak_fallback_key_material(path: Path) -> _KeyMaterial:
    """Load legacy v1 key material for migration only.

    New stores never use this deterministic derivation. Keeping the reader
    allows existing encrypted credentials to be upgraded without signing users
    out.
    """

    salt = _load_or_create_salt(weak_fallback_salt_path(path))
    identity = f"{platform.node()}\0{getpass.getuser()}".encode()
    kdf = Scrypt(salt=salt, length=_AES_KEY_BYTES, n=2**14, r=8, p=1)
    return _KeyMaterial(KEY_SOURCE_WEAK_FALLBACK, kdf.derive(identity))


def _filesystem_key_material(path: Path) -> _KeyMaterial:
    key_path = filesystem_master_key_path(path)
    key = (
        _read_filesystem_master_key(key_path)
        if key_path.exists()
        else _create_filesystem_master_key(key_path)
    )
    return _KeyMaterial(KEY_SOURCE_FILESYSTEM, key)


def _read_filesystem_master_key(path: Path) -> bytes:
    try:
        key = path.read_bytes()
    except OSError as exc:
        raise McpTokenStoreUnavailableError(
            f"Failed to read filesystem OAuth credential key: {path}"
        ) from exc
    if len(key) != _AES_KEY_BYTES:
        raise McpTokenStoreCorruptError(
            f"Filesystem OAuth credential key has invalid length: {path}"
        )
    _set_restrictive_permissions(path)
    return key


def _create_filesystem_master_key(path: Path) -> bytes:
    """Create a fully-written random key without replacing a racing writer.

    The temporary file is fsynced before an atomic hard link publishes it.
    If another process wins the race, its completed key is loaded instead.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(_AES_KEY_BYTES)
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        if os.name != "nt":
            os.fchmod(fd, 0o600)
        handle = os.fdopen(fd, "wb")
        fd = -1
        with handle:
            handle.write(key)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError:
            return _read_filesystem_master_key(path)
        except OSError as exc:
            raise McpTokenStoreUnavailableError(
                f"Failed to create filesystem OAuth credential key: {path}"
            ) from exc
        _set_restrictive_permissions(path)
        _fsync_dir(path.parent)
        return key
    finally:
        if fd >= 0:
            os.close(fd)
        with suppress(FileNotFoundError):
            temp_path.unlink()


def _load_or_create_salt(path: Path) -> bytes:
    if path.exists():
        try:
            salt = path.read_bytes()
        except OSError as exc:
            raise McpTokenStoreUnavailableError(
                f"Failed to read OAuth token-store salt: {path}"
            ) from exc
        if len(salt) < 16:
            raise McpTokenStoreCorruptError(f"OAuth token-store salt is invalid: {path}")
        _set_restrictive_permissions(path)
        return salt
    salt = secrets.token_bytes(32)
    try:
        _secure_atomic_write_bytes(path, salt)
    except OSError as exc:
        raise McpTokenStoreUnavailableError(
            f"Failed to write OAuth token-store salt: {path}"
        ) from exc
    return salt


def _dpapi_key_material(path: Path) -> _KeyMaterial:
    key_path = dpapi_master_key_path(path)
    if key_path.exists():
        try:
            protected = key_path.read_bytes()
        except OSError as exc:
            raise McpTokenStoreUnavailableError(
                f"Failed to read DPAPI MCP OAuth key: {key_path}"
            ) from exc
        try:
            key = _dpapi_unprotect(protected)
        except Exception as exc:  # noqa: BLE001
            raise McpTokenStoreUnavailableError("Failed to unprotect DPAPI MCP OAuth key.") from exc
        if len(key) != _AES_KEY_BYTES:
            raise McpTokenStoreCorruptError(f"DPAPI MCP OAuth key has invalid length: {key_path}")
        _set_restrictive_permissions(key_path)
        return _KeyMaterial(KEY_SOURCE_DPAPI, key)
    key = secrets.token_bytes(_AES_KEY_BYTES)
    try:
        protected = _dpapi_protect(key)
        _secure_atomic_write_bytes(key_path, protected)
    except Exception as exc:  # noqa: BLE001
        raise McpTokenStoreUnavailableError("Failed to create DPAPI MCP OAuth key.") from exc
    return _KeyMaterial(KEY_SOURCE_DPAPI, key)


def _platform_system() -> str:
    return platform.system()


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.c_uint32),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def _dpapi_protect(data: bytes) -> bytes:
    if not _is_windows_os():
        raise McpTokenStoreUnavailableError("DPAPI is only available on Windows.")
    return _crypt_protect_data(data, protect=True)


def _dpapi_unprotect(data: bytes) -> bytes:
    if not _is_windows_os():
        raise McpTokenStoreUnavailableError("DPAPI is only available on Windows.")
    return _crypt_protect_data(data, protect=False)


def _crypt_protect_data(data: bytes, *, protect: bool) -> bytes:
    try:
        crypt32, kernel32 = _load_windows_crypto_api()
    except Exception as exc:  # noqa: BLE001
        raise McpTokenStoreUnavailableError("Failed to load Windows DPAPI.") from exc
    in_buffer = ctypes.create_string_buffer(data)
    in_blob = _DataBlob(len(data), ctypes.cast(in_buffer, ctypes.POINTER(ctypes.c_ubyte)))
    out_blob = _DataBlob()
    call = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    ok = call(
        ctypes.byref(in_blob),
        None,
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(out_blob),
    )
    if not ok:
        raise McpTokenStoreUnavailableError("Windows DPAPI operation failed.")
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        if out_blob.pbData:
            kernel32.LocalFree(out_blob.pbData)


def _load_windows_crypto_api() -> tuple[Any, Any]:
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _configure_windows_crypto_api(crypt32, kernel32)
    return crypt32, kernel32


def _configure_windows_crypto_api(crypt32: Any, kernel32: Any) -> None:
    data_blob_pointer = ctypes.POINTER(_DataBlob)
    crypt32.CryptProtectData.argtypes = [
        data_blob_pointer,
        ctypes.c_wchar_p,
        data_blob_pointer,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        data_blob_pointer,
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [
        data_blob_pointer,
        ctypes.POINTER(ctypes.c_wchar_p),
        data_blob_pointer,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        data_blob_pointer,
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p


def _is_windows_os() -> bool:
    return os.name == "nt"


def _secure_atomic_write_bytes(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        if os.name != "nt":
            os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _set_restrictive_permissions(path, mode=mode)
        _fsync_dir(path.parent)
    finally:
        with suppress(FileNotFoundError):
            temp_path.unlink()


def _set_restrictive_permissions(path: Path, *, mode: int = 0o600) -> None:
    if os.name == "nt":
        return
    with suppress(OSError):
        os.chmod(path, mode)
